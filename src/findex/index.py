import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.models import DocMeta, Posting
from findex.tokenize import tokenize

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--positions", action="store_true")
    args = ap.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    postings, metas = build_index(
        iter_documents(args.root), with_positions=args.positions
    )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()

    print(f"documents: {len(metas)}")
    print(f"terms:     {len(postings)}")
    print(f"build:     {elapsed:.2f}s")
    print(f"peak mem:  {peak / 2**20:.1f} MiB")
    # TODO: save(postings, metas, args.out) після storage.py


if __name__ == "__main__":
    main()