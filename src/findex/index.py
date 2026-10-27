import argparse
import logging
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.models import DocMeta, Posting
from findex.storage import load, save
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

Postings = dict[str, list[Posting]]


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

        metas.append(DocMeta(doc.doc_id, str(doc.path), doc.path.stem, length))

    return dict(postings), metas  # dict(): щоб пошук по відсутньому терміну не створював записів


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс як словник «термін -> список Posting».

    Від Mapping достатньо реалізувати __getitem__, __len__, __iter__ —
    keys/items/values/get/__contains__/__eq__ abc.Mapping дає сам.
    """

    def __init__(self, postings: dict[str, list[Posting]], metas: list[DocMeta]):
        self._postings = postings
        self._metas = metas
        self.closed = False

    # --- конструктори -------------------------------------------------
    @classmethod
    def build(cls, docs: Iterable[Document], *, with_positions: bool = False) -> "Index":
        return cls(*build_index(docs, with_positions=with_positions))

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        return cls(*load(Path(path)))

    def save(self, path: str | Path, fmt: str = "pickle") -> None:
        save(self._postings, self._metas, Path(path), fmt)

    # --- протокол Mapping ----------------------------------------------
    def __getitem__(self, term: str) -> list[Posting]:
        self._check_open()
        return self._postings[term]  # KeyError, як у dict

    def __len__(self) -> int:
        self._check_open()
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        self._check_open()
        return iter(self._postings)

    def __contains__(self, term: object) -> bool:
        # Mapping уміє це сам через try/except KeyError, але так швидше
        self._check_open()
        return term in self._postings

    def __repr__(self) -> str:
        if self.closed:
            return "<Index closed>"
        return (
            f"Index(docs={self.num_docs}, terms={len(self)}, "
            f"avg_len={self.avg_doc_length:.1f}, positions={self.has_positions})"
        )

    # --- статистики корпусу --------------------------------------------
    @property
    def metas(self) -> list[DocMeta]:
        return self._metas

    @property
    def num_docs(self) -> int:
        return len(self._metas)

    @cached_property
    def avg_doc_length(self) -> float:
        # рахується один раз, далі береться з self.__dict__
        return sum(m.length for m in self._metas) / self.num_docs if self._metas else 0.0

    @cached_property
    def has_positions(self) -> bool:
        first = next(iter(self._postings.values()), None)
        return bool(first) and first[0].positions is not None

    def df(self, term: str) -> int:
        """Document frequency: у скількох документах є термін."""
        return len(self.get(term, ()))

    # --- ресурси --------------------------------------------------------
    def close(self) -> None:
        self._postings = {}
        self._metas = []
        # скидаємо закешовані значення cached_property
        self.__dict__.pop("avg_doc_length", None)
        self.__dict__.pop("has_positions", None)
        self.closed = True

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("operation on closed index")


@contextmanager
def open_index(path: Path) -> Iterator[Index]:
    """with open_index(path) as idx: ... — індекс закривається навіть при винятку."""
    index = Index.load(path)
    log.info("opened %s: %r", path, index)
    try:
        yield index
    finally:
        index.close()
        log.info("closed %s", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--format", choices=["pickle", "json"], default="pickle")
    args = ap.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    index = Index.build(iter_documents(args.root), with_positions=args.positions)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()

    print(repr(index))
    print(f"documents: {index.num_docs}")
    print(f"terms:     {len(index)}")
    print(f"build:     {elapsed:.2f}s")
    print(f"peak mem:  {peak / 2**20:.1f} MiB")
   
    t1 = time.perf_counter()
    index.save(args.out, args.format)
    save_time = time.perf_counter() - t1
    size = args.out.stat().st_size / 2**20
    print(f"saved:     {args.out} ({size:.1f} MiB, {save_time:.2f}s)")


if __name__ == "__main__":
    main()