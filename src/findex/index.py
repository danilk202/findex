import logging
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import numpy as np

from findex.arrays import EMPTY, IntArray, PostingArrays
from findex.corpus import Document
from findex.models import DocMeta, Posting
from findex.storage import Format, Postings, load_arrays, save, save_arrays
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


def build_index(
    docs: Iterable[Document], *, with_positions: bool = False
) -> tuple[Postings, list[DocMeta]]:
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    metas: list[DocMeta] = []

    for doc in docs:
        if with_positions:
            positions: defaultdict[str, list[int]] = defaultdict(list)
            length = 0
            for i, tok in enumerate(tokenize(doc.text)):
                positions[tok].append(i)
                length = i + 1
            for term, pos in positions.items():
                postings[term].append(Posting(doc.doc_id, len(pos), tuple(pos)))
        else:
            counts = Counter(tokenize(doc.text))
            length = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append(Posting(doc.doc_id, tf))

        metas.append(DocMeta(doc.doc_id, str(doc.path), doc.title or doc.path.stem, length))

    return dict(postings), metas  # dict(): щоб пошук по відсутньому терміну не створював записів


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс як словник «термін -> список Posting».

    Від Mapping достатньо реалізувати __getitem__, __len__, __iter__ —
    keys/items/values/get/__contains__/__eq__ abc.Mapping дає сам.
    Лаба 8: усередині — масиви NumPy (PostingArrays), а список Posting
    збирається з них лише на вимогу.
    """

    def __init__(self, postings: Postings | PostingArrays, metas: list[DocMeta]) -> None:
        if not isinstance(postings, PostingArrays):
            postings = PostingArrays.from_postings(postings, [m.length for m in metas])
        self._arrays = postings
        self._metas = metas
        self.closed = False

    # --- конструктори -------------------------------------------------
    @classmethod
    @timed
    def build(cls, docs: Iterable[Document], *, with_positions: bool = False) -> "Index":
        return cls(*build_index(docs, with_positions=with_positions))

    @classmethod
    @timed
    def load(cls, path: str | Path) -> "Index":
        return cls(*load_arrays(Path(path)))

    def save(self, path: str | Path, fmt: Format = "pickle") -> None:
        match fmt:
            case "pickle":
                save_arrays(self._arrays, self._metas, Path(path))
            case "json":
                save(self._arrays.to_postings(), self._metas, Path(path), "json")

    # --- протокол Mapping ----------------------------------------------
    def __getitem__(self, term: str) -> list[Posting]:
        self._check_open()
        if term not in self._arrays:
            raise KeyError(term)  # як у dict
        return self._arrays.postings(term)

    def __len__(self) -> int:
        self._check_open()
        return len(self._arrays)

    def __iter__(self) -> Iterator[str]:
        self._check_open()
        return iter(self._arrays)

    def __contains__(self, term: object) -> bool:
        self._check_open()
        return term in self._arrays

    def __repr__(self) -> str:
        if self.closed:
            return "<Index closed>"
        return (
            f"Index(docs={self.num_docs}, terms={len(self)}, "
            f"avg_len={self.avg_doc_length:.1f}, positions={self.has_positions})"
        )

    # --- статистики корпусу --------------------------------------------
    @property
    def arrays(self) -> PostingArrays:
        self._check_open()
        return self._arrays

    @property
    def metas(self) -> list[DocMeta]:
        return self._metas

    @property
    def num_docs(self) -> int:
        return len(self._metas)

    @property
    def avg_doc_length(self) -> float:
        # Публічно це звичайний @property, а кешує приватний cached_property нижче.
        # Причина — pyright strict: cached_property не вважається сумісним з @property
        # у протоколі CorpusStats (scoring.py), і Index перестав би бути CorpusStats.
        return self._avg_doc_length

    @cached_property
    def _avg_doc_length(self) -> float:
        # рахується один раз, далі береться з self.__dict__
        return sum(m.length for m in self._metas) / self.num_docs if self._metas else 0.0

    @property
    def has_positions(self) -> bool:
        return self._arrays.has_positions and len(self._arrays.doc_ids) > 0

    @property
    def num_postings(self) -> int:
        return len(self._arrays.doc_ids)

    @property
    def num_tokens(self) -> int:
        return sum(m.length for m in self._metas)

    def top_terms(self, n: int) -> list[tuple[str, int, int]]:
        """n найчастіших термів: (терм, df, сумарний tf по корпусу)."""
        a = self._arrays
        df, cf = a.document_frequencies(), a.collection_frequencies()
        # за cf, потім df — обидва за спаданням; при рівності — порядок термів у словнику
        order = np.lexsort((np.arange(len(df)), -df, -cf))[:n]
        return [(a.terms[i], int(df[i]), int(cf[i])) for i in order.tolist()]

    def df(self, term: str) -> int:
        """Document frequency: у скількох документах є термін."""
        return self._arrays.df(term)

    def doc_ids(self, term: str) -> IntArray:
        return self._arrays.doc_ids_of(term)

    # --- ресурси --------------------------------------------------------
    def close(self) -> None:
        self._arrays = PostingArrays((), np.zeros(1, np.int64), EMPTY, EMPTY, EMPTY)
        self._metas = []
        # скидаємо закешоване значення cached_property
        self.__dict__.pop("_avg_doc_length", None)
        self.closed = True

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("operation on closed index")


@contextmanager
def open_index(path: str | Path) -> Generator[Index]:
    """with open_index(path) as idx: ... — індекс закривається навіть при винятку."""
    index = Index.load(path)
    log.info("opened %s: %r", path, index)
    try:
        yield index
    finally:
        index.close()
        log.info("closed %s", path)
