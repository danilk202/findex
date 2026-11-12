# Лаба 8: NumPy-скорер дає той самий рейтинг, що й цикл з лаби 3 — порядок і бали до біта.

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.arrays import PostingArrays
from findex.corpus import Document, iter_documents
from findex.engine import Searcher
from findex.index import Index, build_index
from findex.query import parse
from findex.ranking import SearchResult, search, search_np, top_k
from findex.scoring import BM25, TfIdf

SCORERS = [BM25(), TfIdf(), BM25(k1=0.9, b=0.4)]
DATA = Path(__file__).resolve().parents[1] / "data"
LAB3_QUERIES = [
    "whale OR ship OR sea",
    "monster OR creature",
    "murder OR crime",
    "marriage love",
    "treasure OR pirate",
    "ghost OR spirit",
    "war (soldier OR army)",
    '"time traveller" OR future',
    "river raft",
    "rabbit OR witch",
    "whale ship captain",
    "the",
    "harpooneer",
    "love NOT marriage",
]

words = st.sampled_from(["a", "b", "c", "d", "e", "f"])
docs = st.lists(st.lists(words, min_size=1, max_size=12).map(" ".join), min_size=1, max_size=30)


def make_index(texts: list[str]) -> Index:
    corpus = (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(texts))
    return Index(*build_index(corpus, with_positions=True))


def same(a: list[SearchResult], b: list[SearchResult]) -> None:
    assert [(r.doc_id, r.score) for r in a] == [(r.doc_id, r.score) for r in b]


@settings(max_examples=150, deadline=None)
@given(docs, st.lists(words, min_size=1, max_size=4), st.integers(1, 12), st.booleans())
def test_same_ranking_as_lab3(texts: list[str], terms: list[str], k: int, filtered: bool):
    # маленький словник — багато однакових балів, тобто перевіряється й порядок при рівності
    index = make_index(texts)
    cands = {i for i in range(index.num_docs) if i % 3 != 1} if filtered else None
    for scorer in SCORERS:
        same(search_np(index, terms, scorer, k, cands), search(index, terms, scorer, k, cands))


def test_top_k_ties_break_by_larger_doc_id():
    scores = np.array([1.0, 2.0, 2.0, 2.0, 0.5])
    pool = np.arange(5, dtype=np.int32)
    assert top_k(scores, pool, 2).tolist() == [3, 2]
    assert top_k(scores, pool, 4).tolist() == [3, 2, 1, 0]
    assert top_k(scores, pool, 10).tolist() == [3, 2, 1, 0, 4]
    assert top_k(scores, pool[:0], 3).tolist() == []


def test_candidates_without_terms_get_zero(small_index: Index):
    # як у лабі 3: кандидат без жодного терма запиту (запит із самого NOT) має бал 0
    got = search_np(small_index, ["dog"], BM25(), 5, candidates={0, 1})
    assert [r.doc_id for r in got] == [1, 0]
    assert got[1].score == 0.0


def test_unknown_terms_and_empty_index():
    index = make_index(["a b"])
    assert search_np(index, ["zzz"], BM25()) == []
    assert search_np(index, [], BM25(), candidates={0})[0].score == 0.0


def test_arrays_roundtrip_postings(small_index: Index):
    postings = small_index.arrays.to_postings()
    again = PostingArrays.from_postings(postings, [m.length for m in small_index.metas])
    assert again.to_postings() == postings
    assert small_index.arrays.df("the") == 2
    assert small_index.arrays.has_doc("dog", 1) and not small_index.arrays.has_doc("dog", 0)
    assert small_index.top_terms(2) == [("the", 2, 3), ("sat", 2, 2)]


@pytest.fixture(scope="module")
def full_index() -> Index:
    return Index(*build_index(iter_documents(DATA), with_positions=True))


@pytest.mark.skipif(not DATA.is_dir(), reason="повний корпус лежить лише локально (data/)")
@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_full_corpus_same_as_lab3(full_index: Index, scorer: str):
    index = full_index
    s = BM25() if scorer == "bm25" else TfIdf()
    numpy_searcher, lab3 = Searcher(index, s), Searcher(index, s, backend="python")
    for q in LAB3_QUERIES:
        same(numpy_searcher.search(q, k=25), lab3.search(q, k=25))
        same(search_np(index, parse(q).terms(), s, 10), search(index, parse(q).terms(), s, 10))
