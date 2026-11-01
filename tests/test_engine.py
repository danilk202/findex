import logging

import pytest

from findex.corpus import iter_documents
from findex.engine import Searcher
from findex.index import Index
from findex.scoring import BM25


@pytest.fixture
def searcher(tmp_path):
    texts = {
        "d0": "python async await event loop",
        "d1": "python java loop event",
        "d2": "java coffee",
    }
    for name, text in texts.items():
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    return Searcher(Index.build(iter_documents(tmp_path), with_positions=True), BM25())


def test_ranked_boolean(searcher):
    res = searcher.search('python NOT java "event loop"')
    assert [r.title for r in res] == ["d0"]


def test_not_only_query(searcher):
    assert [r.title for r in searcher.search("NOT python")] == ["d2"]


def test_repeat_is_cache_hit(searcher, caplog):
    with caplog.at_level(logging.INFO, logger="findex.cache"):
        first = searcher.search("python async")
        second = searcher.search("python async")
    assert first == second
    assert caplog.text.count("cache miss") == 1
    assert caplog.text.count("cache hit") == 1


def test_same_tree_shares_cache_entry(searcher):
    searcher.search("python async")
    searcher.search("python  AND   async")  # інший рядок, те саме дерево
    info = searcher.match.cache_info()
    assert (info.hits, info.misses) == (1, 1)


def test_search_is_timed(searcher, caplog):
    with caplog.at_level(logging.INFO, logger="findex.timing"):
        searcher.search("python")
    assert "Searcher.search took" in caplog.text
