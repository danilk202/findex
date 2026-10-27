import heapq
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from findex.index import Index
from findex.scoring import Scorer


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    score: float
    doc_id: int
    title: str = field(compare=False)


def search(
    index: Index, terms: Iterable[str], scorer: Scorer, k: int = 10
) -> list[SearchResult]:

    scores: defaultdict[int, float] = defaultdict(float)

  
    for term in terms:
        postings = index.get(term)
        if not postings:
            continue
        df = len(postings)
        for p in postings:
            doc_len = index.metas[p.doc_id].length
            scores[p.doc_id] += scorer.score(p.tf, df, doc_len, index)

            
    results = (SearchResult(score, doc_id, index.metas[doc_id].title) for doc_id, score in scores.items())
    return heapq.nlargest(k, results)

  
