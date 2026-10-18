from collections.abc import Callable
from typing import Literal

from findex.models import Posting
from findex.tokenize import tokenize

IdList = list[int]
Engine = Literal["merge", "set"]


def and_merge(a: IdList, b: IdList) -> IdList:
    i = j = 0
    out: IdList = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def or_merge(a: IdList, b: IdList) -> IdList:
    i = j = 0
    out: IdList = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def andnot_merge(a: IdList, b: IdList) -> IdList:
    """Документи з a, яких немає в b."""
    i = j = 0
    out: IdList = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


def and_set(a: IdList, b: IdList) -> IdList:
    return sorted(set(a) & set(b))


def or_set(a: IdList, b: IdList) -> IdList:
    return sorted(set(a) | set(b))


def andnot_set(a: IdList, b: IdList) -> IdList:
    return sorted(set(a) - set(b))


_OPS: dict[str, tuple[Callable[..., IdList], ...]] = {
    "merge": (and_merge, or_merge, andnot_merge),
    "set": (and_set, or_set, andnot_set),
}


def parse(query: str) -> list[tuple[list[str], list[str]]]:
    """'a b NOT c OR d' -> [(['a','b'], ['c']), (['d'], [])]"""
    groups: list[tuple[list[str], list[str]]] = []
    pos: list[str] = []
    neg: list[str] = []
    negate = False
    for word in query.split():
        if word == "OR":
            groups.append((pos, neg))
            pos, neg = [], []
        elif word == "NOT":
            negate = True
        else:
            for term in tokenize(word):
                (neg if negate else pos).append(term)
            negate = False
    groups.append((pos, neg))
    return [g for g in groups if g[0] or g[1]]


def _ids(postings: dict[str, list[Posting]], term: str) -> IdList:
    return [p.doc_id for p in postings.get(term, ())]


def evaluate(
    query: str,
    postings: dict[str, list[Posting]],
    n_docs: int,
    engine: Engine = "merge",
) -> IdList:
    and_, or_, andnot = _OPS[engine]
    universe = list(range(n_docs))
    result: IdList = []
    for pos, neg in parse(query):
        lists = sorted((_ids(postings, t) for t in pos), key=len)  # рідкісні першими
        cur = lists[0] if lists else universe
        for lst in lists[1:]:
            cur = and_(cur, lst)
        for t in neg:
            cur = andnot(cur, _ids(postings, t))
        result = or_(result, cur)
    return result