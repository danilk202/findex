import argparse
import gc
import sys
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.models import Posting
from findex.tokenize import tokenize


@dataclass(frozen=True)  # без slots: кожен екземпляр має власний __dict__
class PlainPosting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] | None = None


def build(root: Path, variant: str):
    if variant == "array":
        index = defaultdict(lambda: array("I"))
    else:
        index = defaultdict(list)
    make = {"plain": PlainPosting, "slots": Posting}.get(variant)
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            if make:
                index[term].append(make(doc.doc_id, tf))
            else:
                index[term].extend((doc.doc_id, tf))  # пара doc_id, tf підряд
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("variant", choices=["plain", "slots", "array"])
    args = ap.parse_args()

    tracemalloc.start()
    index = build(args.root, args.variant)
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n = sum(len(v) for v in index.values())
    if args.variant == "array":
        n //= 2
    print(f"variant:   {args.variant}")
    print(f"postings:  {n}")
    print(f"index:     {current / 2**20:.1f} MiB ({current / n:.0f} bytes/posting)")
    print(f"peak:      {peak / 2**20:.1f} MiB")
    if args.variant != "array":
        p = index["the"][0]
        print(f"getsizeof(one posting): {sys.getsizeof(p)} bytes")
        if hasattr(p, "__dict__"):
            print(f"getsizeof(its __dict__): {sys.getsizeof(p.__dict__)} bytes")


if __name__ == "__main__":
    main()