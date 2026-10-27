import pytest

from findex.corpus import iter_documents
from findex.index import Index
from findex.ranking import SearchResult, search
from findex.scoring import BM25, TfIdf


@pytest.fixture
def toy(tmp_path):
    # той самий іграшковий корпус, що ми рахували руками
    (tmp_path / "d0.txt").write_text("python async await python", encoding="utf-8")
    (tmp_path / "d1.txt").write_text("python java", encoding="utf-8")
    (tmp_path / "d2.txt").write_text("java java java coffee", encoding="utf-8")
    return Index.build(iter_documents(tmp_path))


def test_search_result_ordering():
    low = SearchResult(0.4, 1, "d1")
    high = SearchResult(1.9, 0, "d0")
    assert low < high
    assert max(low, high) is high
    assert SearchResult(1.0, 5, "a") == SearchResult(1.0, 5, "інша назва")  # title не порівнюється


def test_tfidf_hand_numbers(toy):
    res = search(toy, ["python", "async"], TfIdf())
    assert [r.title for r in res] == ["d0", "d1"]  # d2 немає жодного терма
    assert res[0].score == pytest.approx(1.910, abs=1e-3)
    assert res[1].score == pytest.approx(0.405, abs=1e-3)


def test_missing_term_is_ignored(toy):
    assert search(toy, ["python", "zebra"], TfIdf()) == search(toy, ["python"], TfIdf())


def test_k_limits_results(toy):
    assert len(search(toy, ["python", "java"], BM25(), k=1)) == 1
    assert len(search(toy, ["python", "java"], BM25(), k=10)) == 3


def test_results_sorted_desc(toy):
    res = search(toy, ["java", "coffee", "python"], BM25())
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)
    assert res[0].title == "d2"  # java×3 + рідкісне coffee


def test_scorers_interchangeable(toy):
    # одна функція search, різні скорери — порядок може відрізнятись, набір документів той самий
    a = {r.doc_id for r in search(toy, ["python", "async"], TfIdf())}
    b = {r.doc_id for r in search(toy, ["python", "async"], BM25())}
    assert a == b == {0, 1}
