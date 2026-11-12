import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from findex.index import Index
from findex.query import Node, parse
from findex.ranking import SearchResult, search, search_np
from findex.scoring import Scorer
from findex.timing import timed

if TYPE_CHECKING:
    # Тип, який повертає lru_cache (з методами cache_info/cache_clear), у typeshed
    # має лише приватне ім'я. Імпорт потрібен тільки для анотації, у рантаймі його нема.
    from functools import _lru_cache_wrapper  # pyright: ignore[reportPrivateUsage]

log = logging.getLogger("findex.cache")

type Matcher = "_lru_cache_wrapper[frozenset[int]]"


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


type Backend = Literal["numpy", "python"]


class Searcher:
    def __init__(
        self, index: Index, scorer: Scorer, cache_size: int = 256, backend: Backend = "numpy"
    ):
        self.index = index
        self.scorer = scorer
        self.match = make_matcher(index, cache_size)
        # python — скорер з лаби 3 (цикл по Posting), numpy — лаба 8; результат однаковий
        self.rank = search_np if backend == "numpy" else search

    @timed
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        tree = parse(query)
        log.debug("parsed %r -> %s", query, tree)
        hits_before = self.match.cache_info().hits
        ids = self.match(tree)
        if self.match.cache_info().hits > hits_before:
            log.info("cache hit:  %s", tree)
        log.debug("%d candidate documents", len(ids))
        return self.rank(self.index, tree.terms(), self.scorer, k, candidates=ids)
