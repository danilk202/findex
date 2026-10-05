import argparse
import time
import tracemalloc
from collections import Counter
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()

    docs = list(islice(iter_documents(args.root), args.limit))
    tokenized = [list(tokenize(d.text)) for d in docs]
    counts = Counter(tok for toks in tokenized for tok in toks)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()

    print(f"documents: {len(docs)}")
    print(f"tokens:    {sum(counts.values())}")
    print(f"vocab:     {len(counts)}")
    print(f"elapsed:   {elapsed:.2f}s")
    print(f"peak mem:  {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()