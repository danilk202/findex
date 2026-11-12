import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from findex.engine import Searcher
from findex.index import open_index
from findex.query import QuerySyntaxError
from findex.scoring import BM25, Scorer, TfIdf
from findex.web.service import Mode

SCORERS: dict[str, Scorer] = {"TF-IDF": TfIdf(), "BM25": BM25()}


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int = 5) -> float:

    return sum(1 for title in ranked[:k] if title in relevant) / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("queries", type=Path, nargs="+", help="один або кілька файлів з розміткою")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--details", action="store_true", help="показати топ-k для кожного запиту")
    ap.add_argument("--emb", type=Path, help="лаба 8: порівняти BM25, semantic і hybrid")
    args = ap.parse_args()

    qrels: list[dict[str, Any]] = []
    for path in args.queries:
        for q in json.loads(path.read_text(encoding="utf-8")):
            qrels.append(q | {"kind": path.stem})
    if args.emb:
        modes_table(args.index, qrels, args.emb, args.k, args.details)
        return
    k = args.k
    names = list(SCORERS)
    rows: list[tuple[str, int, list[float], float]] = []
    details: list[str] = []

    with open_index(args.index) as index:
        searchers = {n: Searcher(index, s) for n, s in SCORERS.items()}
        for q in qrels:
            relevant = set(q["relevant"])
            scores: list[float] = []
            for n in names:
                top = [r.title for r in searchers[n].search(q["query"], k=k)]
                scores.append(precision_at_k(top, relevant, k))
                marked = ", ".join(f"**{t}**" if t in relevant else t for t in top)
                details.append(f"- `{q['query']}` {n}: {marked}")
            best_possible = min(len(relevant), k) / k
            rows.append((q["query"], len(relevant), scores, best_possible))

    print(f"| Запит | Релевантних | {' | '.join(f'{n} P@{k}' for n in names)} | Макс. P@{k} |")
    print("|---|---|" + "---|" * len(names) + "---|")
    for query, n_rel, scores, best in rows:
        cells = " | ".join(f"{s:.2f}" for s in scores)
        print(f"| `{query}` | {n_rel} | {cells} | {best:.2f} |")
    means = " | ".join(f"**{mean(r[2][i] for r in rows):.2f}**" for i in range(len(names)))
    print(f"| **Середнє** | | {means} | {mean(r[3] for r in rows):.2f} |")

    if args.details:
        print(f"\nТоп-{k} (релевантні жирним):")
        print("\n".join(details))


MODES: dict[str, Mode] = {"BM25": "lexical", "Семантичний": "semantic", "Гібрид (RRF)": "hybrid"}


def modes_table(index_path: Path, qrels: list[dict[str, Any]], emb: Path, k: int, details: bool):
    from findex.semantic import SemanticSearch
    from findex.web.service import SearchService

    rows: list[tuple[str, str, int, list[float], float]] = []
    lines: list[str] = []
    with open_index(index_path) as index:
        service = SearchService(index, semantic=SemanticSearch.open(emb, index))
        for q in qrels:
            relevant = set(q["relevant"])
            scores: list[float] = []
            for name, mode in MODES.items():
                try:
                    hits = service.search(q["query"], "bm25", k, 1, mode).hits
                except QuerySyntaxError:
                    hits = ()
                top = [index.metas[h.doc_id].title for h in hits]
                scores.append(precision_at_k(top, relevant, k))
                marked = ", ".join(f"**{t}**" if t in relevant else t for t in top) or "—"
                lines.append(f"- `{q['query']}` {name}: {marked}")
            rows.append((q["query"], q["kind"], len(relevant), scores, min(len(relevant), k) / k))

    names = list(MODES)
    print(f"| Запит | Набір | Релевантних | {' | '.join(f'{n} P@{k}' for n in names)} | Макс. |")
    print("|---|---|---:|" + "---:|" * len(names) + "---:|")
    for query, kind, n_rel, scores, best in rows:
        cells = " | ".join(f"{s:.2f}" for s in scores)
        print(f"| `{query}` | {kind} | {n_rel} | {cells} | {best:.2f} |")
    for kind in dict.fromkeys(r[1] for r in rows):
        part = [r for r in rows if r[1] == kind]
        means = " | ".join(f"**{mean(r[3][i] for r in part):.2f}**" for i in range(len(names)))
        print(f"| **Середнє: {kind}** | | | {means} | {mean(r[4] for r in part):.2f} |")
    if details:
        print(f"\nТоп-{k} (релевантні жирним):")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
