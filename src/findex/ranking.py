import heapq
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from findex.index import Index
from findex.scoring import Scorer


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
        SearchResult(scores.get(doc_id, 0.0), doc_id, index.metas[doc_id].title)
        for doc_id in pool
    )
    return heapq.nlargest(k, results)
