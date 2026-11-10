# Pydantic — лише на межі: перевірка того, що прийшло, і форма того, що йде назовні.
# Усередині (service.py, index.py) — dataclass-и з лаб 2–3; перетворення тут, у from_*.

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from findex.scoring import ScorerName
from findex.web.service import Hit, IndexStats, Segments
from findex.web.service import SearchPage as SearchPageData


class SearchParams(BaseModel):
    # extra="forbid": /search?q=x&lmit=5 — це 422, а не мовчки проігнорований параметр
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    q: str = Field(min_length=1, max_length=500, description='Напр.: whale OR ship "white whale"')
    k: int = Field(default=10, ge=1, le=50, description="Результатів на сторінку")
    scorer: ScorerName = "bm25"
    page: int = Field(default=1, ge=1, le=100)


class DocParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=5000, ge=1, le=50_000, description="Скільки символів тексту")


def _plain(snippet: Segments) -> tuple[str, list[tuple[int, int]]]:
    # сегменти -> звичайний текст + [початок, кінець) кожного підсвіченого терма
    text: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for piece, hit in snippet:
        if hit:
            spans.append((pos, pos + len(piece)))
        text.append(piece)
        pos += len(piece)
    return "".join(text), spans


class HitOut(BaseModel):
    rank: int
    doc_id: int
    title: str
    score: float
    tokens: int
    snippet: str
    highlights: list[tuple[int, int]] = Field(description="Межі підсвічених термів у snippet")

    @classmethod
    def from_hit(cls, hit: Hit) -> Self:
        text, spans = _plain(hit.snippet)
        return cls(
            rank=hit.rank,
            doc_id=hit.doc_id,
            title=hit.title,
            score=round(hit.score, 4),
            tokens=hit.tokens,
            snippet=text,
            highlights=spans,
        )


class SearchOut(BaseModel):
    query: str
    tree: str = Field(description="Як запит розібрано")
    scorer: ScorerName
    page: int
    k: int
    total: int = Field(description="Скільки документів підходить під запит")
    pages: int
    took_ms: float
    results: list[HitOut]

    @classmethod
    def from_page(cls, data: SearchPageData) -> Self:
        return cls(
            query=data.query,
            tree=data.tree,
            scorer=data.scorer,
            page=data.page,
            k=data.k,
            total=data.total,
            pages=data.pages,
            took_ms=round(data.took_ms, 2),
            results=[HitOut.from_hit(h) for h in data.hits],
        )


class DocOut(BaseModel):
    doc_id: int
    title: str
    path: str
    tokens: int
    chars: int
    offset: int
    text: str


class TermOut(BaseModel):
    term: str
    df: int
    cf: int = Field(description="Скільки разів трапляється в усьому корпусі")


class StatsOut(BaseModel):
    documents: int
    terms: int
    postings: int
    tokens: int
    avg_doc_length: float
    positions: bool
    top_terms: list[TermOut]

    @classmethod
    def from_stats(cls, s: IndexStats) -> Self:
        return cls(
            documents=s.documents,
            terms=s.terms,
            postings=s.postings,
            tokens=s.tokens,
            avg_doc_length=s.avg_doc_length,
            positions=s.positions,
            top_terms=[TermOut(term=t, df=df, cf=cf) for t, df, cf in s.top_terms],
        )


class HealthOut(BaseModel):
    status: Literal["ok", "loading", "error"]
    documents: int | None = None
    detail: str | None = None


class ErrorOut(BaseModel):
    detail: str
    request_id: str
