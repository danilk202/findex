import heapq
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from findex.index import Index
from findex.scoring import FloatArray, IntArray, Scorer

type BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    """order=True генерує __lt__, __gt__ ... — порівняння йде по полях зверху вниз:
    спершу score, при рівності doc_id. title у порівнянні не бере участі."""

    score: float
    doc_id: int
    title: str = field(compare=False)


def search(
    index: Index,
    terms: Iterable[str],
    scorer: Scorer,
    k: int = 10,
    candidates: Collection[int] | None = None,
) -> list[SearchResult]:
    """Ранжований пошук: бал документа = сума балів усіх термів запиту, що в ньому є.

    candidates — множина doc_id, яку вже відібрав Boolean-фільтр (дерево запиту).
    Тоді бали рахуються лише для них, а кандидат без жодного терма отримує 0
    (так працює запит із самого NOT: `NOT python`).
    """
    scores: defaultdict[int, float] = defaultdict(float)

    for term in terms:
        postings = index.get(term)
        if not postings:
            continue
        df = len(postings)
        for p in postings:
            if candidates is not None and p.doc_id not in candidates:
                continue
            doc_len = index.metas[p.doc_id].length
            scores[p.doc_id] += scorer.score(p.tf, df, doc_len, index)

    pool = scores if candidates is None else candidates
    results = (
        SearchResult(scores.get(doc_id, 0.0), doc_id, index.metas[doc_id].title) for doc_id in pool
    )
    return heapq.nlargest(k, results)


# ------------------------------------------------------------------ лаба 8: NumPy


def top_k(scores: FloatArray, pool: IntArray, k: int) -> IntArray:
    # k найкращих doc_id з pool у тому ж порядку, що й heapq.nlargest над SearchResult:
    # бал за спаданням, при рівних балах — більший doc_id вище.
    if k <= 0 or len(pool) == 0:
        return pool[:0]
    s = scores[pool]
    if len(pool) > k:
        # argpartition — O(n): k найбільших опиняються спереду, але в довільному порядку,
        # і серед рівних на межі вибір теж довільний. Межу добираємо явно.
        kth = s[np.argpartition(-s, k - 1)[:k]].min()
        above = pool[s > kth]
        ties = np.sort(pool[s == kth])[::-1][: k - len(above)]
        pool = np.concatenate((above, ties))
        s = scores[pool]
    return pool[np.lexsort((-pool, -s))]


def term_scores(index: Index, terms: Iterable[str], scorer: Scorer) -> tuple[FloatArray, BoolArray]:
    # бали всіх документів одразу: на кожен терм — кілька операцій над масивами
    arr = index.arrays
    scores = np.zeros(index.num_docs, dtype=np.float64)
    touched = np.zeros(index.num_docs, dtype=np.bool_)
    for term in terms:
        a, b = arr.span(term)
        if a == b:
            continue
        ids = arr.doc_ids[a:b]
        scores[ids] += scorer.score_array(arr.tfs[a:b], b - a, arr.lengths[ids], index)
        touched[ids] = True
    return scores, touched


def search_np(
    index: Index,
    terms: Iterable[str],
    scorer: Scorer,
    k: int = 10,
    candidates: Collection[int] | None = None,
) -> list[SearchResult]:
    # Те саме, що search() вище, але бали — виразами NumPy, а топ-k — через argpartition.
    scores, touched = term_scores(index, terms, scorer)
    if candidates is None:
        pool = np.flatnonzero(touched).astype(np.int32)
    else:
        pool = np.fromiter(candidates, dtype=np.int32, count=len(candidates))
    best = top_k(scores, pool, k)
    metas = index.metas
    ids: list[int] = best.tolist()
    return [SearchResult(float(scores[d]), d, metas[d].title) for d in ids]
