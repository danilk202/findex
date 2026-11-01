import logging
from collections.abc import Callable
from functools import lru_cache

from findex.index import Index
from findex.query import Node, parse
from findex.ranking import SearchResult, search
from findex.scoring import Scorer
from findex.timing import timed

log = logging.getLogger("findex.cache")

Matcher = Callable[[Node], frozenset[int]]


def make_matcher(index: Index, maxsize: int = 256) -> Matcher:
    """Замикання над index + lru_cache: дерево запиту -> множина doc_id.

    Чому не просто @lru_cache на функцію (index, tree)? Index — це Mapping, а
    Mapping не хешується, тож бути частиною ключа кешу він не може. Замикання
    «вшиває» index усередину, і ключем стає лише дерево (frozen dataclass — хешується).
    """

    @lru_cache(maxsize=maxsize)
    def match(tree: Node) -> frozenset[int]:
        log.info("cache miss: %s", tree)
        # frozenset: кешований результат віддається всім викликачам — він має бути незмінним
        return frozenset(tree.evaluate(index))

    return match


class Searcher:
    def __init__(self, index: Index, scorer: Scorer, cache_size: int = 256):
        self.index = index
        self.scorer = scorer
        self.match = make_matcher(index, cache_size)

    @timed
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        tree = parse(query)
        hits_before = self.match.cache_info().hits
        ids = self.match(tree)
        if self.match.cache_info().hits > hits_before:
            log.info("cache hit:  %s", tree)
        return search(self.index, tree.terms(), self.scorer, k, candidates=ids)
