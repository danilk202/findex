import argparse
import math
from pathlib import Path

from findex.index import Index, open_index
from findex.scoring import BM25, TfIdf


def check_rare_vs_frequent(index: Index, rare: str, frequent: str) -> None:
    bm25, tfidf = BM25(), TfIdf()
    print(f"\n## 1. Рідкісний термін важить більше за частий ({rare} vs {frequent})\n")
    print("| Терм | df | idf TF-IDF | idf BM25 | BM25, tf=1, документ середньої довжини |")
    print("|---|---|---|---|---|")
    avg = round(index.avg_doc_length)
    for term in (rare, frequent):
        df = index.df(term)
        print(
            f"| {term} | {df} | {tfidf.score(1, df, avg, index):.3f} "
            f"| {_idf_bm25(index, df):.3f} "
            f"| {bm25.score(1, df, avg, index):.3f} |"
        )


def _idf_bm25(index: Index, df: int) -> float:
    return math.log(1 + (index.num_docs - df + 0.5) / (df + 0.5))


def check_saturation(index: Index, term: str, title: str) -> None:
    bm25, tfidf = BM25(), TfIdf()
    meta = next(m for m in index.metas if m.title == title)
    df = index.df(term)
    real_tf = next(p.tf for p in index[term] if p.doc_id == meta.doc_id)
    print(f"\n## 2. Насичення tf: «{term}» у {title} ({meta.length} токенів, df={df})\n")
    print("| tf | BM25 | внесок саме tf-го входження | TF-IDF |")
    print("|---|---|---|---|")
    for tf in sorted({1, 2, 5, 10, 19, 20, real_tf}):
        s = bm25.score(tf, df, meta.length, index)
        marginal = s - (bm25.score(tf - 1, df, meta.length, index) if tf > 1 else 0.0)
        note = " (реальний tf)" if tf == real_tf else ""
        print(
            f"| {tf}{note} | {s:.3f} | {marginal:+.4f} "
            f"| {tfidf.score(tf, df, meta.length, index):.1f} |"
        )
    one = bm25.score(1, df, meta.length, index)
    gain = bm25.score(20, df, meta.length, index) - bm25.score(19, df, meta.length, index)
    print(f"\n20-те повторення додає {gain:.4f}, тобто {gain / one:.1%} від внеску першого.")


def check_length(index: Index) -> None:
    """Шукає терм, що трапляється рівно 1 раз і в найкоротшій, і в найдовшій книзі."""
    bm25, tfidf = BM25(), TfIdf()
    short = min(index.metas, key=lambda m: m.length)
    long = max(index.metas, key=lambda m: m.length)
    candidates: list[tuple[int, str]] = []
    for term, postings in index.items():
        tfs = {p.doc_id: p.tf for p in postings}
        if tfs.get(short.doc_id) == 1 and tfs.get(long.doc_id) == 1:
            candidates.append((len(postings), term))
    if not candidates:
        print(f"\nУ корпусі немає терма з tf=1 і в {short.title}, і в {long.title}.")
        return
    df, term = min(candidates)  # найрідкісніший з таких — найнаочніший
    print(f"\n## 3. Коротка книга з одним входженням вище довгої («{term}», df={df})\n")
    print("| Книга | Токенів | tf | BM25 | TF-IDF |")
    print("|---|---|---|---|---|")
    for m in (short, long):
        b = bm25.score(1, df, m.length, index)
        t = tfidf.score(1, df, m.length, index)
        print(f"| {m.title} | {m.length} | 1 | {b:.3f} | {t:.3f} |")
    ratio = bm25.score(1, df, short.length, index) / bm25.score(1, df, long.length, index)
    print(f"\nBM25 ставить {short.title} у {ratio:.1f} раза вище; TF-IDF їх не розрізняє.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("--rare", default="whale")
    ap.add_argument("--frequent", default="captain")
    ap.add_argument("--saturation-term", default="whale")
    ap.add_argument("--saturation-doc", default="moby")
    args = ap.parse_args()
    with open_index(args.index) as index:
        check_rare_vs_frequent(index, args.rare, args.frequent)
        check_saturation(index, args.saturation_term, args.saturation_doc)
        check_length(index)


if __name__ == "__main__":
    main()
