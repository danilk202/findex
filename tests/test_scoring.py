import math
from types import SimpleNamespace

import pytest

from findex.scoring import BM25, Scorer, TfIdf


TOY = SimpleNamespace(num_docs=3, avg_doc_length=10 / 3)




def test_tfidf_hand_calculation():
    s = TfIdf()
    python_d0 = s.score(tf=2, df=2, doc_len=4, stats=TOY)
    async_d0 = s.score(tf=1, df=1, doc_len=4, stats=TOY)
    python_d1 = s.score(tf=1, df=2, doc_len=2, stats=TOY)
    assert python_d0 == pytest.approx(0.811, abs=1e-3)
    assert async_d0 == pytest.approx(1.099, abs=1e-3)
    assert python_d0 + async_d0 == pytest.approx(1.910, abs=1e-3)  
    assert python_d1 == pytest.approx(0.405, abs=1e-3)              


def test_tfidf_word_everywhere_scores_zero():
    assert TfIdf().score(tf=5, df=3, doc_len=4, stats=TOY) == 0


def test_tfidf_is_linear_in_tf():
    s = TfIdf()
    one = s.score(1, 2, 4, TOY)
    assert s.score(200, 2, 4, TOY) == pytest.approx(200 * one)



STATS = SimpleNamespace(num_docs=25, avg_doc_length=100)


def test_bm25_rare_term_beats_frequent():
    s = BM25()
    assert s.score(1, df=1, doc_len=100, stats=STATS) > s.score(1, df=20, doc_len=100, stats=STATS)


def test_bm25_idf_never_negative():
    
    assert BM25().score(1, df=25, doc_len=100, stats=STATS) > 0


def test_bm25_tf_saturates():
    s = BM25(b=0)  
    one = s.score(1, 5, 100, STATS)
    assert s.score(20, 5, 100, STATS) / one == pytest.approx(2.33, abs=0.01)
    assert s.score(200, 5, 100, STATS) / one == pytest.approx(2.48, abs=0.01)
    # 20-те повторення додає менше 2% від внеску першого
    gain_20th = s.score(20, 5, 100, STATS) - s.score(19, 5, 100, STATS)
    assert gain_20th < 0.02 * one


def test_bm25_short_doc_beats_long():
    s = BM25()
    short = s.score(1, 5, doc_len=50, stats=STATS)
    long = s.score(1, 5, doc_len=1000, stats=STATS)
    assert short / long == pytest.approx(6.5, abs=0.1)


def test_bm25_b_zero_ignores_length():
    s = BM25(b=0)
    assert s.score(1, 5, 50, STATS) == s.score(1, 5, 1000, STATS)


def test_bm25_defaults():
    assert BM25() == BM25(k1=1.5, b=0.75)




def test_scorers_are_interchangeable():
    scorers: list[Scorer] = [TfIdf(), BM25(), BM25(k1=1.2, b=0.5)]
    for s in scorers:
        assert math.isfinite(s.score(3, 2, 4, TOY))
