# Лаба 8: постінги як масиви NumPy замість мільйонів об'єктів Posting.
#
# Формат CSR (як у scipy.sparse): усі постінги всіх термів лежать підряд в одному масиві,
# а indptr[r]..indptr[r+1] — це шматок терма з номером r. Так само позиції:
# pos_indptr[i]..pos_indptr[i+1] — позиції i-го постінга в масиві positions.
#
#   terms      ("the", "whale", ...)            порядок як у словнику з build_index
#   indptr     int64[terms + 1]
#   doc_ids    int32[postings]                  у межах терма — за зростанням
#   tfs        int32[postings]
#   lengths    int32[docs]                      довжина кожного документа в токенах
#   pos_indptr int64[postings + 1] | None       None — індекс без позицій
#   positions  int32[tokens] | None

from collections.abc import Iterator, Sequence
from itertools import chain
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from findex.models import Posting

type Postings = dict[str, list[Posting]]
type IntArray = npt.NDArray[np.int32]
type OffsetArray = npt.NDArray[np.int64]

EMPTY: Final[IntArray] = np.zeros(0, dtype=np.int32)


class PostingArrays:
    __slots__ = (
        "doc_ids",
        "indptr",
        "lengths",
        "pos_indptr",
        "positions",
        "rows",
        "terms",
        "tfs",
    )

    def __init__(
        self,
        terms: Sequence[str],
        indptr: OffsetArray,
        doc_ids: IntArray,
        tfs: IntArray,
        lengths: IntArray,
        pos_indptr: OffsetArray | None = None,
        positions: IntArray | None = None,
    ) -> None:
        if len(indptr) != len(terms) + 1 or len(doc_ids) != len(tfs):
            raise ValueError("inconsistent posting arrays")
        if (pos_indptr is None) != (positions is None):
            raise ValueError("pos_indptr and positions go together")
        self.terms = tuple(terms)
        self.rows = {t: r for r, t in enumerate(self.terms)}
        self.indptr = indptr
        self.doc_ids = doc_ids
        self.tfs = tfs
        self.lengths = lengths
        self.pos_indptr = pos_indptr
        self.positions = positions

    # ------------------------------------------------------------ побудова

    @classmethod
    def from_postings(cls, postings: Postings, lengths: Sequence[int]) -> "PostingArrays":
        # Один прохід по об'єктах Posting (результат build_index / merge / json),
        # далі в пам'яті лишаються лише масиви.
        terms = list(postings)
        lists = list(postings.values())
        counts = np.fromiter((len(pl) for pl in lists), dtype=np.int64, count=len(lists))
        indptr = np.zeros(len(lists) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        n = int(indptr[-1])
        flat = list(chain.from_iterable(lists))
        doc_ids = np.fromiter((p.doc_id for p in flat), dtype=np.int32, count=n)
        tfs = np.fromiter((p.tf for p in flat), dtype=np.int32, count=n)

        pos_indptr: OffsetArray | None = None
        positions: IntArray | None = None
        if flat and flat[0].positions is not None:
            sizes = np.fromiter((len(p.positions or ()) for p in flat), dtype=np.int64, count=n)
            pos_indptr = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(sizes, out=pos_indptr[1:])
            positions = np.fromiter(
                chain.from_iterable(p.positions or () for p in flat),
                dtype=np.int32,
                count=int(pos_indptr[-1]),
            )
        return cls(
            terms,
            indptr,
            doc_ids,
            tfs,
            np.asarray(lengths, dtype=np.int32),
            pos_indptr,
            positions,
        )

    def to_postings(self) -> Postings:
        return {t: self.postings(t) for t in self.terms}

    # ------------------------------------------------------- збереження

    def state(self) -> dict[str, Any]:
        # Для pickle: лише кортеж рядків і масиви. Масив NumPy pickle пише як один
        # суцільний буфер байтів — без виклику Python-коду на кожен елемент.
        return {
            "terms": self.terms,
            "indptr": self.indptr,
            "doc_ids": self.doc_ids,
            "tfs": self.tfs,
            "lengths": self.lengths,
            "pos_indptr": self.pos_indptr,
            "positions": self.positions,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "PostingArrays":
        return cls(
            state["terms"],
            state["indptr"],
            state["doc_ids"],
            state["tfs"],
            state["lengths"],
            state["pos_indptr"],
            state["positions"],
        )

    # ------------------------------------------------------------ доступ

    @property
    def num_docs(self) -> int:
        return len(self.lengths)

    @property
    def has_positions(self) -> bool:
        return self.positions is not None

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[str]:
        return iter(self.terms)

    def __contains__(self, term: object) -> bool:
        return term in self.rows

    def span(self, term: str) -> tuple[int, int]:
        # [a, b) постінгів терма в doc_ids/tfs; (0, 0) — терма немає
        r = self.rows.get(term)
        if r is None:
            return 0, 0
        return int(self.indptr[r]), int(self.indptr[r + 1])

    def doc_ids_of(self, term: str) -> IntArray:
        a, b = self.span(term)
        return self.doc_ids[a:b]  # зріз — це view, без копіювання

    def tfs_of(self, term: str) -> IntArray:
        a, b = self.span(term)
        return self.tfs[a:b]

    def df(self, term: str) -> int:
        a, b = self.span(term)
        return b - a

    def has_doc(self, term: str, doc_id: int) -> bool:
        ids = self.doc_ids_of(term)
        i = int(np.searchsorted(ids, doc_id))
        return i < len(ids) and int(ids[i]) == doc_id

    def positions_in(self, term: str, doc_id: int) -> IntArray:
        # позиції терма в одному документі (порожньо, якщо його там нема)
        if self.pos_indptr is None or self.positions is None:
            return EMPTY
        a, b = self.span(term)
        i = a + int(np.searchsorted(self.doc_ids[a:b], doc_id))
        if i >= b or int(self.doc_ids[i]) != doc_id:
            return EMPTY
        return self.positions[self.pos_indptr[i] : self.pos_indptr[i + 1]]

    def postings(self, term: str) -> list[Posting]:
        # Сумісність з лабами 2–3: список Posting, зібраний з масивів на льоту.
        a, b = self.span(term)
        ids = self.doc_ids[a:b].tolist()
        tfs = self.tfs[a:b].tolist()
        if self.pos_indptr is None or self.positions is None:
            return [Posting(d, t) for d, t in zip(ids, tfs, strict=True)]
        bounds = self.pos_indptr[a : b + 1].tolist()
        pos = self.positions
        return [
            Posting(d, t, tuple(pos[bounds[i] : bounds[i + 1]].tolist()))
            for i, (d, t) in enumerate(zip(ids, tfs, strict=True))
        ]

    def document_frequencies(self) -> OffsetArray:
        return np.diff(self.indptr)

    def collection_frequencies(self) -> OffsetArray:
        # сумарний tf кожного терма; reduceat по початках шматків
        if len(self.tfs) == 0:
            return np.zeros(len(self.terms), dtype=np.int64)
        starts = np.minimum(self.indptr[:-1], len(self.tfs) - 1)
        cf = np.add.reduceat(self.tfs.astype(np.int64), starts)
        cf[self.document_frequencies() == 0] = 0  # reduceat на порожньому шматку дає не 0
        return cf

    def nbytes(self) -> int:
        arrays = (
            self.indptr,
            self.doc_ids,
            self.tfs,
            self.lengths,
            self.pos_indptr,
            self.positions,
        )
        return sum(a.nbytes for a in arrays if a is not None)
