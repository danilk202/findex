import argparse
import time
import tracemalloc
from pathlib import Path

from findex.boolean import evaluate
from findex.storage import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("query")
    ap.add_argument("--engine", choices=["merge", "set"], default="merge")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    postings, metas = load(args.index)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    ids = evaluate(args.query, postings, len(metas), args.engine)
    search_time = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for doc_id in ids[: args.limit]:
        m = metas[doc_id]
        print(f"{m.doc_id:>3}  {m.title}  ({m.length} tokens)")
    print(f"found:     {len(ids)} documents (showing {min(len(ids), args.limit)})")
    print(f"load:      {load_time:.2f}s")
    print(f"search:    {search_time * 1000:.2f}ms ({args.engine})")
    print(f"peak mem:  {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()