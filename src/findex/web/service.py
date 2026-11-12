"""Пошук для вебу: тонкий шар над Index/Searcher з лаб 2–3, без жодного FastAPI."""

import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Final, Literal

from findex.corpus import read_text
from findex.engine import Searcher
from findex.index import Index
from findex.models import DocMeta
from findex.query import parse
from findex.scoring import SCORER_NAMES, ScorerName, make_scorer
from findex.semantic import (
    STOPWORDS,
    SemanticSearch,
    SemanticUnavailableError,
    query_text,
    rrf,
)
from findex.snippets import best_window, find_hits, make_snippet
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

# Маркери підсвітки, як у CLI: символів \x02/\x03 у книжковому тексті не буває.
MARKS: Final = ("\x02", "\x03")
CHUNK: Final = 20_000  # символів на одну сторінку перегляду документа
HYBRID_DEPTH: Final = 100  # скільки місць кожного рейтингу йде в RRF

type Mode = Literal["lexical", "semantic", "hybrid"]
MODES: Final[tuple[Mode, ...]] = ("lexical", "semantic", "hybrid")

# Шапка книжки Gutenberg: «Title: Moby Dick; Or, The Whale» / «Author: Herman Melville»
_HEADER = re.compile(r"^(Title|Author):[ \t]*(.+?)[ \t]*$", re.MULTILINE)

type TextSource = Callable[[DocMeta], str]
type Segments = tuple[tuple[str, bool], ...]  # (шматок тексту, чи це підсвічений терм)


def _read(meta: DocMeta) -> str:
    return read_text(Path(meta.path))


def segments(marked: str) -> Segments:
    # "a \x02whale\x03 b" -> (("a ", False), ("whale", True), (" b", False))
    out: list[tuple[str, bool]] = []
    for i, piece in enumerate(marked.split(MARKS[0])):
        if i == 0:
            out.append((piece, False))
            continue
        hit, _, rest = piece.partition(MARKS[1])
        out += [(hit, True), (rest, False)]
    return tuple((text, hit) for text, hit in out if text)


@dataclass(frozen=True, slots=True)
class Hit:
    rank: int
    doc_id: int
    title: str
    author: str | None
    score: float
    tokens: int
    snippet: Segments


@dataclass(frozen=True, slots=True)
class SearchPage:
    query: str
    tree: str
    scorer: ScorerName
    page: int
    k: int
    total: int  # скільки документів пройшли Boolean-фільтр
    hits: tuple[Hit, ...]
    took_ms: float
    mode: Mode = "lexical"

    @property
    def pages(self) -> int:
        return max(1, math.ceil(self.total / self.k))


@dataclass(frozen=True, slots=True)
class DocChunk:
    meta: DocMeta
    title: str
    author: str | None
    part: int
    parts: int
    offset: int
    chars: int  # довжина всього тексту
    text: Segments
    focus: int | None = None


@dataclass(frozen=True, slots=True)
class IndexStats:
    documents: int
    terms: int
    postings: int
    tokens: int
    avg_doc_length: float
    positions: bool
    top_terms: tuple[tuple[str, int, int], ...]


class SearchService:
    def __init__(
        self,
        index: Index,
        texts: TextSource = _read,
        *,
        snippet_width: int = 80,
        text_cache: int = 64,
        semantic: SemanticSearch | None = None,
    ) -> None:
        self.index = index
        self.semantic = semantic
        self.snippet_width = snippet_width
        # Searcher на кожен скорер: у кожного свій lru_cache «дерево -> doc_id»
        self._searchers = {name: Searcher(index, make_scorer(name)) for name in SCORER_NAMES}

        @lru_cache(maxsize=text_cache)
        def text_of(doc_id: int) -> str:
            # 25 книг ≈ 17 МБ: читати файл на кожен сніпет дорожче, ніж тримати в пам'яті
            return texts(index.metas[doc_id])

        self.text_of = text_of
        self._headings: dict[int, tuple[str, str | None]] = {}

    @property
    def num_docs(self) -> int:
        return self.index.num_docs

    def search(
        self,
        query: str,
        scorer: ScorerName = "bm25",
        k: int = 10,
        page: int = 1,
        mode: Mode = "lexical",
    ) -> SearchPage:
        # QuerySyntaxError / NeedsPositionsError / SemanticUnavailableError летять нагору —
        # їх перекладає маршрут
        t0 = time.perf_counter()
        depth = k * page
        fragments: dict[int, tuple[int, int]] = {}
        if mode == "lexical":
            searcher = self._searchers[scorer]
            tree = parse(query)
            matched = searcher.match(tree)
            ranked = [
                (r.doc_id, r.score)
                for r in searcher.rank(self.index, tree.terms(), searcher.scorer, depth, matched)
            ]
            terms, shown, total = list(tree.terms()), str(tree), len(matched)
        else:
            semantic = self._need_semantic()
            hits = semantic.search(query, depth if mode == "semantic" else HYBRID_DEPTH)
            fragments = {h.doc_id: (h.start, h.end) for h in hits}
            terms = [t for t in tokenize(query_text(query)) if t not in STOPWORDS]
            if mode == "semantic":
                ranked = [(h.doc_id, h.score) for h in hits]
                shown, total = f"≈ {query_text(query)}", len(semantic.index.present)
            else:
                # гібрид: місця в рейтингу BM25 (з Boolean-фільтром) і в семантичному -> RRF
                searcher = self._searchers[scorer]
                tree = parse(query)
                matched = searcher.match(tree)
                lexical = searcher.rank(
                    self.index, tree.terms(), searcher.scorer, HYBRID_DEPTH, matched
                )
                lex_ids = [r.doc_id for r in lexical]
                ranked = rrf([lex_ids, [h.doc_id for h in hits]])
                terms, shown, total = list(tree.terms()), f"{tree} + ≈ semantic", len(ranked)
                # документ знайдено за словами — сніпет зі словами; інакше — фрагмент
                fragments = {d: span for d, span in fragments.items() if d not in matched}
        start = (page - 1) * k
        out = tuple(
            Hit(
                rank=start + i,
                doc_id=doc_id,
                title=self.heading(doc_id)[0],
                author=self.heading(doc_id)[1],
                score=score,
                tokens=self.index.metas[doc_id].length,
                snippet=self.snippet(doc_id, terms, fragments.get(doc_id)),
            )
            for i, (doc_id, score) in enumerate(ranked[start:depth], 1)
        )
        took = (time.perf_counter() - t0) * 1000
        log.info(
            "search %r (%s/%s, page %d): %d hits in %.1f ms", query, mode, scorer, page, total, took
        )
        return SearchPage(query, shown, scorer, page, k, total, out, took, mode)

    @property
    def semantic_available(self) -> bool:
        return self.semantic is not None

    def _need_semantic(self) -> SemanticSearch:
        if self.semantic is None:
            raise SemanticUnavailableError("semantic search is not available on this server")
        return self.semantic

    def heading(self, doc_id: int) -> tuple[str, str | None]:
        # Справжня назва й автор із шапки Gutenberg; інакше — назва з індексу (ім'я файлу).
        if (cached := self._headings.get(doc_id)) is None:
            head = self.text_of(doc_id)[:4000]
            found = {m.group(1): m.group(2) for m in _HEADER.finditer(head)}
            title = found.get("Title") or self.index.metas[doc_id].title.replace("_", " ")
            cached = self._headings[doc_id] = (title, found.get("Author"))
        return cached

    def snippet(
        self, doc_id: int, terms: list[str], fragment: tuple[int, int] | None = None
    ) -> Segments:
        # Лише терми, що справді є в документі (видно з постінгів): тоді make_snippet
        # знаходить вікно з усіма ними і зупиняється рано, а не сканує всю книжку.
        text = self.text_of(doc_id)
        if fragment is not None:
            # семантичний збіг: найкращий фрагмент, у ньому підсвічуються слова запиту, якщо є
            s, e = fragment
            return segments(make_snippet(text[s:e], terms, width=self.snippet_width, mark=MARKS))
        present = [t for t in terms if self.index.arrays.has_doc(t, doc_id)]
        terms = present or terms
        return segments(make_snippet(text, terms, width=self.snippet_width, mark=MARKS))

    def document(
        self, doc_id: int, query: str | None = None, part: int | None = None
    ) -> DocChunk | None:
        # Шматок документа з підсвіченими термами. part=None і є запит -> шматок, де
        # стоїть сніпет із результатів; focus — номер сегмента з цим входженням.
        if not 0 <= doc_id < self.num_docs:
            return None
        meta = self.index.metas[doc_id]
        text = self.text_of(doc_id)
        bounds = _chunk_bounds(text)
        terms = set(parse(query).terms()) if query else set[str]()
        hits, best = best_window(text, terms, self.snippet_width)
        if best is not None:
            hits = find_hits(text, terms)  # best_window міг зупинитись рано — потрібні всі
        focus_at = hits[best][0] if best is not None else None
        if part is None:
            part = 1
            if focus_at is not None:
                part = next(i for i, b in enumerate(bounds[1:], 1) if focus_at < b)
        part = min(max(part, 1), len(bounds) - 1)
        lo, hi = bounds[part - 1], bounds[part]
        pieces: list[tuple[str, bool]] = []
        focus: int | None = None
        cur = lo
        for s, e, _ in hits:
            if s >= lo and e <= hi:
                if s > cur:
                    pieces.append((text[cur:s], False))
                if s == focus_at:
                    focus = len(pieces)
                pieces.append((text[s:e], True))
                cur = e
        if hi > cur:
            pieces.append((text[cur:hi], False))
        title, author = self.heading(doc_id)
        return DocChunk(
            meta=meta,
            title=title,
            author=author,
            part=part,
            parts=len(bounds) - 1,
            offset=lo,
            chars=len(text),
            text=tuple(pieces),
            focus=focus,
        )

    def raw_text(self, doc_id: int, offset: int, limit: int) -> tuple[DocMeta, str, int] | None:
        if not 0 <= doc_id < self.num_docs:
            return None
        text = self.text_of(doc_id)
        return self.index.metas[doc_id], text[offset : offset + limit], len(text)

    @cached_property
    def stats(self) -> IndexStats:
        # рахується один раз: num_postings і top_terms проходять увесь словник
        idx = self.index
        return IndexStats(
            documents=idx.num_docs,
            terms=len(idx),
            postings=idx.num_postings,
            tokens=idx.num_tokens,
            avg_doc_length=round(idx.avg_doc_length, 1),
            positions=idx.has_positions,
            top_terms=tuple(idx.top_terms(10)),
        )


def _chunk_bounds(text: str) -> list[int]:
    # Межі шматків по ~CHUNK символів, зсунуті до кінця рядка, щоб не різати слова.
    bounds = [0]
    while bounds[-1] + CHUNK < len(text):
        nl = text.find("\n", bounds[-1] + CHUNK)
        if nl == -1:
            break
        bounds.append(nl + 1)
    if bounds[-1] != len(text) or len(bounds) == 1:
        bounds.append(len(text))
    return bounds
