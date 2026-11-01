import argparse
import gc
import sys
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from findex.corpus import iter_documents
from findex.models import Posting
from findex.tokenize import tokenize


@dataclass(frozen=True)  # без slots: кожен екземпляр має власний __dict__
class PlainPosting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] | None = None


type Variant = Literal["plain", "slots", "array"]
type AnyPosting = PlainPosting | Posting


def build_objects(root: Path, variant: Literal["plain", "slots"]) -> dict[str, list[AnyPosting]]:
    make = PlainPosting if variant == "plain" else Posting
    index: defaultdict[str, list[AnyPosting]] = defaultdict(list)
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            index[term].append(make(doc.doc_id, tf))
    return index


def build_array(root: Path) -> dict[str, array[int]]:
    index: defaultdict[str, array[int]] = defaultdict(lambda: array("I"))
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            index[term].extend((doc.doc_id, tf))  # пара doc_id, tf підряд
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("variant", choices=["plain", "slots", "array"])
    args = ap.parse_args()

    variant: Variant = args.variant
    tracemalloc.start()
    objects: dict[str, list[AnyPosting]] = {}
    if variant == "array":
        arrays = build_array(args.root)
        n = sum(len(v) for v in arrays.values()) // 2
    else:
        objects = build_objects(args.root, variant)
        n = sum(len(v) for v in objects.values())
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"variant:   {args.variant}")
    print(f"postings:  {n}")
    print(f"index:     {current / 2**20:.1f} MiB ({current / n:.0f} bytes/posting)")
    print(f"peak:      {peak / 2**20:.1f} MiB")
    if objects:
        p = objects["the"][0]
        print(f"getsizeof(one posting): {sys.getsizeof(p)} bytes")
        if hasattr(p, "__dict__"):
            print(f"getsizeof(its __dict__): {sys.getsizeof(p.__dict__)} bytes")


if __name__ == "__main__":
    main()
