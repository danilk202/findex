"""CLI пошуку.

    uv run python -m findex.search index.bin 'python AND (async OR await) NOT java "event loop"'
    uv run python -m findex.search index.bin "whale" "whale" -v    # другий раз — cache hit у лозі
    uv run python -m findex.search index.bin                       # інтерактивний режим
    uv run python -m findex.search index.bin "king NOT queen" --engine merge   # Boolean з лаби 2
"""

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from findex.boolean import evaluate
from findex.engine import Searcher
from findex.index import Index, open_index
from findex.query import QuerySyntaxError, parse
from findex.scoring import BM25, Scorer, TfIdf
from findex.snippets import ANSI, MARKDOWN, doc_snippet

SCORERS: dict[str, Scorer] = {"tfidf": TfIdf(), "bm25": BM25()}


def read_queries(given: list[str]) -> Iterator[str]:
    if given:
        yield from given
        return
    while True:  # інтерактивно: порожній рядок або Ctrl+Z/Ctrl+D — вихід
        try:
            q = input("query> ").strip()
        except EOFError:
            return
        if not q:
            return
        yield q


def run_ranked(searcher: Searcher, query: str, args: argparse.Namespace) -> None:
    results = searcher.search(query, k=args.limit)
    terms = list(parse(query).terms())
    mark = ANSI if sys.stdout.isatty() else MARKDOWN
    print(f"\n{query}  [{args.scorer}]")
    if not results:
        print("  nothing found")
    for rank, r in enumerate(results, 1):
        m = searcher.index.metas[r.doc_id]
        print(f"{rank:>3}. {r.score:7.3f}  {m.title}  ({m.length} tokens)")
        if not args.no_snippets:
            print(f"       {doc_snippet(m.path, terms, width=args.width, mark=mark)}")


def run_legacy(index: Index, query: str, engine: str, limit: int) -> None:
    ids = evaluate(query, index, index.num_docs, engine)  # лаба 2, без рейтингу
    print(f"\n{query}  [boolean/{engine}]")
    for doc_id in ids[:limit]:
        m = index.metas[doc_id]
        print(f"  {m.doc_id:>3}  {m.title}  ({m.length} tokens)")
    print(f"  found: {len(ids)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("queries", nargs="*", help="без запитів — інтерактивний режим")
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="bm25")
    ap.add_argument("-k", "--limit", type=int, default=10)
    ap.add_argument("--width", type=int, default=80, help="півширина сніпета")
    ap.add_argument("--no-snippets", action="store_true")
    ap.add_argument("--engine", choices=["merge", "set"], help="старий Boolean-пошук з лаби 2")
    ap.add_argument("-v", "--verbose", action="store_true", help="лог @timed і кешу")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="  [%(name)s] %(message)s",
    )
    sys.stdout.reconfigure(errors="replace")  # Windows-консоль не любить деякі символи

    with open_index(args.index) as index:
        searcher = Searcher(index, SCORERS[args.scorer])
        for query in read_queries(args.queries):
            try:
                if args.engine:
                    run_legacy(index, query, args.engine, args.limit)
                else:
                    run_ranked(searcher, query, args)
            except (QuerySyntaxError, ValueError) as e:
                print(f"\n{query}\n  error: {e}")


if __name__ == "__main__":
    main()
