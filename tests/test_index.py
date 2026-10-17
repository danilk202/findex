from findex.corpus import iter_documents
from findex.index import build_index


def make_corpus(tmp_path):
    (tmp_path / "a.txt").write_text("the cat sat on the mat", encoding="utf-8")
    (tmp_path / "b.txt").write_text("the dog sat", encoding="utf-8")
    return tmp_path


def test_postings_sorted_and_tf(tmp_path):
    postings, metas = build_index(iter_documents(make_corpus(tmp_path)))
    assert [(p.doc_id, p.tf) for p in postings["the"]] == [(0, 2), (1, 1)]
    assert [p.doc_id for p in postings["sat"]] == [0, 1]
    assert [m.length for m in metas] == [6, 3]


def test_positions_flag(tmp_path):
    postings, _ = build_index(
        iter_documents(make_corpus(tmp_path)), with_positions=True
    )
    assert postings["the"][0].positions == (0, 4)


def test_missing_term_not_created(tmp_path):
    postings, _ = build_index(iter_documents(make_corpus(tmp_path)))
    assert "zebra" not in postings
    assert postings.get("zebra") is None