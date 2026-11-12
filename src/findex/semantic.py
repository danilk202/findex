# Лаба 8: семантичний пошук. Документ ділиться на фрагменти, кожен фрагмент — вектор
# змісту (ембединг) довжини 1, усі вектори — одна матриця у .npy. Запит теж стає вектором,
# і косинусна близькість до всіх фрагментів — це одне множення матриці на вектор.

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from findex.index import Index
from findex.ranking import top_k

log = logging.getLogger(__name__)

type FloatArray = npt.NDArray[np.float32]

# Ваги sentence-transformers/all-MiniLM-L6-v2 у форматі ONNX (експорт Xenova), зафіксовані
# ревізією і sha256: те, чим рахувалась матриця, і те, чим сервер кодує запит, — один файл.
MODEL_NAME: Final = "all-MiniLM-L6-v2"
MODEL_REPO: Final = "Xenova/all-MiniLM-L6-v2"
MODEL_REVISION: Final = "751bff37182d3f1213fa05d7196b954e230abad9"
MODEL_FILES: Final = {
    "onnx/model.onnx": "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e",
    "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
}
DIM: Final = 384
MAX_TOKENS: Final = 256  # довші фрагменти модель обрізає
FRAGMENT_WORDS: Final = 150  # ~200 токенів WordPiece — влазить у 256
RRF_K: Final = 60  # константа з оригінальної статті про Reciprocal Rank Fusion

_WORDS = re.compile(r"\S+")
# у сніпеті семантичного збігу підсвічуємо слова запиту, але не службові: «a» в кожному рядку
# fmt: off
STOPWORDS: Final = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "by", "with",
    "from", "into", "as", "is", "are", "was", "were", "be", "been", "it", "its", "his", "her",
    "their", "this", "that", "these", "those", "he", "she", "they", "i", "you", "we", "not",
    "no", "s",
})
# fmt: on
_OPERATORS = re.compile(r'\b(?:AND|OR|NOT)\b|[()"]')


class SemanticUnavailableError(RuntimeError):
    """Немає матриці ембедингів або моделі — семантичний режим вимкнено."""


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> FloatArray: ...


# ---------------------------------------------------------------- модель


def default_model_dir() -> Path:
    env = os.environ.get("FINDEX_MODEL_DIR")
    return Path(env) if env else Path.home() / ".cache" / "findex" / MODEL_NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_model(directory: Path | None = None) -> Path:
    # Качає модель з Hugging Face (один раз), перевіряє sha256 кожного файлу.
    import httpx

    directory = directory or default_model_dir()
    for name, sha in MODEL_FILES.items():
        path = directory / name
        if path.is_file() and _sha256(path) == sha:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}"
        log.info("downloading %s", url)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
        except httpx.HTTPError as e:
            raise SemanticUnavailableError(f"cannot download {url}: {e}") from e
        if (got := _sha256(tmp)) != sha:
            tmp.unlink()
            raise SemanticUnavailableError(f"{name}: sha256 {got} != {sha}")
        tmp.replace(path)
    return directory


class MiniLM:
    # Токенізатор (WordPiece) + трансформер в onnxruntime + mean pooling + нормалізація —
    # те саме, що робить sentence-transformers, але без torch (~100 МБ замість ~2 ГБ).
    def __init__(self, directory: Path, threads: int | None = None) -> None:
        # у onnxruntime і tokenizers немає повних типів — працюємо з ними як з Any
        import onnxruntime  # pyright: ignore[reportMissingTypeStubs]
        import tokenizers

        ort: Any = onnxruntime
        tokenizer_cls: Any = tokenizers.Tokenizer
        model, tok = directory / "onnx/model.onnx", directory / "tokenizer.json"
        if not model.is_file() or not tok.is_file():
            raise SemanticUnavailableError(f"model not found in {directory} (findex embed)")
        self.tokenizer: Any = tokenizer_cls.from_file(str(tok))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session: Any = ort.InferenceSession(
            str(model), opts, providers=["CPUExecutionProvider"]
        )
        self.inputs: set[str] = {str(i.name) for i in self.session.get_inputs()}

    @property
    def name(self) -> str:
        return MODEL_NAME

    def encode(self, texts: Sequence[str]) -> FloatArray:
        enc = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = np.asarray(self.session.run(None, feeds)[0], dtype=np.float32)
        # mean pooling: середнє векторів токенів без паддінгу (broadcasting маски по осі 384)
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        return normalize(pooled.astype(np.float32))


def normalize(vectors: FloatArray) -> FloatArray:
    # Довжина 1: тоді скалярний добуток і є косинусом кута між векторами
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)


# ---------------------------------------------------------------- фрагменти


def fragments(text: str, words: int = FRAGMENT_WORDS) -> list[tuple[int, int]]:
    # [початок, кінець) символів кожного шматка по `words` слів, без перекриття
    spans = [m.span() for m in _WORDS.finditer(text)]
    return [
        (spans[i][0], spans[min(i + words, len(spans)) - 1][1]) for i in range(0, len(spans), words)
    ]


def query_text(query: str) -> str:
    # мова запитів (AND/OR/NOT, дужки, лапки) моделі не потрібна — лише слова
    return " ".join(_OPERATORS.sub(" ", query).split())


def corpus_fingerprint(index: Index) -> str:
    # Матриця прив'язана до текстів: інший корпус (оновлена книжка на Gutenberg) — інші
    # межі фрагментів. Відбиток — довжини документів у токенах.
    lengths = ",".join(str(m.length) for m in index.metas)
    return hashlib.sha256(lengths.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- збереження


def is_fresh(emb: Path, index: Index) -> bool:
    try:
        meta = json.loads((emb / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return meta.get("corpus") == corpus_fingerprint(index) and (emb / "vectors.npy").is_file()


@dataclass(frozen=True, slots=True)
class EmbedReport:
    fragments: int
    seconds: float
    path: Path


def build_embeddings(
    index: Index,
    texts: Callable[[int], str],
    embedder: Embedder,
    out: Path,
    *,
    words: int = FRAGMENT_WORDS,
    batch: int = 16,
    on_batch: Callable[[int, int], None] | None = None,
) -> EmbedReport:
    t0 = time.perf_counter()
    rows: list[tuple[int, int, int]] = []
    pieces: list[str] = []
    for doc_id in range(index.num_docs):
        text = texts(doc_id)
        for s, e in fragments(text, words):
            rows.append((doc_id, s, e))
            pieces.append(text[s:e])
    # схожі за довжиною фрагменти в одному пакеті — менше паддінгу, менше марної роботи
    order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
    vectors = np.zeros((len(pieces), DIM), dtype=np.float32)
    for i in range(0, len(order), batch):
        part = order[i : i + batch]
        vectors[part] = embedder.encode([pieces[j] for j in part])
        if on_batch:
            on_batch(min(i + batch, len(order)), len(order))

    out.mkdir(parents=True, exist_ok=True)
    # float16: 384 × 2 байти на фрагмент; точності для косинуса досить (див. README)
    np.save(out / "vectors.npy", vectors.astype(np.float16))
    np.save(out / "fragments.npy", np.array(rows, dtype=np.int32).reshape(-1, 3))
    meta = {
        "model": embedder.name,
        "revision": MODEL_REVISION,
        "dim": DIM,
        "words": words,
        "documents": index.num_docs,
        "fragments": len(rows),
        "corpus": corpus_fingerprint(index),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return EmbedReport(len(rows), time.perf_counter() - t0, out)


# ---------------------------------------------------------------- пошук


@dataclass(frozen=True, slots=True)
class SemanticHit:
    doc_id: int
    score: float  # косинус найкращого фрагмента
    start: int  # межі цього фрагмента в тексті документа
    end: int


class SemanticIndex:
    def __init__(self, vectors: FloatArray, frags: npt.NDArray[np.int32], num_docs: int) -> None:
        if len(vectors) != len(frags):
            raise ValueError("vectors and fragments differ in length")
        self.vectors = vectors.astype(np.float32)  # матриця фрагментів × 384
        self.doc = frags[:, 0]
        self.spans = frags[:, 1:]
        self.num_docs = num_docs
        # фрагменти лежать за doc_id; starts — де починається кожен документ
        self.present = np.unique(self.doc)
        self.starts = np.searchsorted(self.doc, self.present)

    @classmethod
    def load(cls, path: Path, index: Index) -> "SemanticIndex":
        meta: dict[str, Any] = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("corpus") != corpus_fingerprint(index):
            raise SemanticUnavailableError(f"{path} was built for another corpus")
        vectors = np.load(path / "vectors.npy")
        frags = np.load(path / "fragments.npy")
        return cls(vectors, frags, index.num_docs)

    @property
    def nbytes(self) -> int:
        return self.vectors.nbytes

    def doc_scores(self, qvec: FloatArray) -> FloatArray:
        # Косинус до кожного фрагмента — одне множення (n × 384) @ (384,); документ
        # отримує найкращий свій фрагмент: maximum.reduceat по шматках одного doc_id.
        sims = self.vectors @ qvec.reshape(-1)
        best = np.full(self.num_docs, -np.inf, dtype=np.float32)
        if len(sims):
            best[self.present] = np.maximum.reduceat(sims, self.starts)
        return best

    def best_fragment(self, qvec: FloatArray, doc_id: int) -> tuple[float, int, int]:
        a, b = np.searchsorted(self.doc, [doc_id, doc_id + 1])
        sims = self.vectors[a:b] @ qvec.reshape(-1)
        i = int(np.argmax(sims))
        s, e = self.spans[a + i]
        return float(sims[i]), int(s), int(e)

    def search(self, qvec: FloatArray, k: int) -> list[SemanticHit]:
        scores = self.doc_scores(qvec)
        best = top_k(scores.astype(np.float64), self.present.astype(np.int32), k)
        hits: list[SemanticHit] = []
        for d in best.tolist():
            score, s, e = self.best_fragment(qvec, d)
            hits.append(SemanticHit(d, score, s, e))
        return hits


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    # Reciprocal Rank Fusion: документ на місці r у списку отримує 1 / (k + r).
    # Бали різних моделей (BM25 ~ 0..20, косинус ~ 0..1) не порівнюються, а місця — так.
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class SemanticSearch:
    # Модель + матриця: те, що потрібно сервісу і CLI для mode=semantic|hybrid
    def __init__(self, index: SemanticIndex, embedder: Embedder, cache: int = 256) -> None:
        self.index = index
        self.embedder = embedder
        # той самий запит на наступній сторінці чи в гібридному режимі — без моделі
        self._encode = lru_cache(maxsize=cache)(self._encode_uncached)

    @classmethod
    def open(
        cls, emb: Path, index: Index, model_dir: Path | None = None, threads: int | None = None
    ) -> "SemanticSearch":
        if not (emb / "vectors.npy").is_file():
            raise SemanticUnavailableError(f"no embeddings in {emb} (findex embed INDEX)")
        return cls(
            SemanticIndex.load(emb, index), MiniLM(model_dir or default_model_dir(), threads)
        )

    def _encode_uncached(self, text: str) -> FloatArray:
        return self.embedder.encode([text])[0]

    def encode_query(self, query: str) -> FloatArray:
        return self._encode(query_text(query) or query)

    def search(self, query: str, k: int) -> list[SemanticHit]:
        return self.index.search(self.encode_query(query), k)
