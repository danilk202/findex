import argparse
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()

    n_docs = 0

    def docs() -> Iterator[Document]:
        nonlocal n_docs
        for d in islice(iter_documents(args.root), args.limit):
            n_docs += 1
            yield d

    counts = Counter(tok for d in docs() for tok in tokenize(d.text))

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()

    print(f"documents: {n_docs}")
    print(f"tokens:    {sum(counts.values())}")
    print(f"vocab:     {len(counts)}")
    print(f"elapsed:   {elapsed:.2f}s")
    print(f"peak mem:  {peak / 2**20:.1f} MiB")
    print("top 50:")
    for term, c in counts.most_common(50):
        print(f"{c:>10}  {term}")


if __name__ == "__main__":
    main()
