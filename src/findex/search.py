import argparse
import time
import tracemalloc
from pathlib import Path

from findex.boolean import evaluate
from findex.index import open_index
from findex.ranking import search
from findex.scoring import BM25, Scorer, TfIdf
from findex.tokenize import tokenize

SCORERS: dict[str, Scorer] = {"tfidf": TfIdf(), "bm25": BM25()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("query")
    ap.add_argument("--engine", choices=["merge", "set"], default="merge")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--scorer",
        choices=sorted(SCORERS),
        help="ранжований пошук; без прапорця — Boolean",
    )
    args = ap.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    with open_index(args.index) as index:
        load_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        if args.scorer:
            results = search(
                index, tokenize(args.query), SCORERS[args.scorer], k=args.limit
            )
            mode = args.scorer
        else:
            
            ids = evaluate(args.query, index, index.num_docs, args.engine)
            mode = args.engine
        search_time = time.perf_counter() - t1
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if args.scorer:
            for r in results:
                m = index.metas[r.doc_id]
                print(f"{r.score:8.3f}  {m.doc_id:>3}  {m.title}  ({m.length} tokens)")
            print(f"shown:     top {len(results)}")
        else:
            for doc_id in ids[: args.limit]:
                m = index.metas[doc_id]
                print(f"{m.doc_id:>3}  {m.title}  ({m.length} tokens)")
            print(f"found:     {len(ids)} documents (showing {min(len(ids), args.limit)})")
    print(f"load:      {load_time:.2f}s")
    print(f"search:    {search_time * 1000:.2f}ms ({mode})")
    print(f"peak mem:  {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()