

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from findex.engine import Searcher
from findex.index import open_index
from findex.scoring import BM25, Scorer, TfIdf

SCORERS: dict[str, Scorer] = {"TF-IDF": TfIdf(), "BM25": BM25()}


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int = 5) -> float:
    
    return sum(1 for title in ranked[:k] if title in relevant) / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("queries", type=Path)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--details", action="store_true", help="показати топ-k для кожного запиту")
    args = ap.parse_args()

    qrels = json.loads(args.queries.read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    main()
