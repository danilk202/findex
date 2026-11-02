from pathlib import Path

from findex.corpus import iter_documents
from findex.index import Index, build_index


def test_postings_sorted_and_tf(small_corpus: Path):
    postings, metas = build_index(iter_documents(small_corpus))
    assert [(p.doc_id, p.tf) for p in postings["the"]] == [(0, 2), (1, 1)]
    assert [p.doc_id for p in postings["sat"]] == [0, 1]
    assert [m.length for m in metas] == [6, 3]


def test_positions_flag(small_corpus: Path):
    postings, _ = build_index(iter_documents(small_corpus), with_positions=True)
    assert postings["the"][0].positions == (0, 4)
    assert postings["the"][0].tf == 2


def test_missing_term_not_created(small_index: Index):
    assert "zebra" not in small_index
    assert small_index.get("zebra") is None
    assert "zebra" not in small_index  # get не створив запис


def test_top_terms_and_totals(small_index: Index):
    assert small_index.top_terms(1) == [("the", 2, 3)]
    assert small_index.num_postings == 8
    assert small_index.num_tokens == 9


def test_bad_encoding_is_replaced(make_corpus, caplog):
    root = make_corpus({"ok": "fine text"})
    (root / "bad.txt").write_bytes(b"caf\xe9 latin-1")  # не UTF-8
    docs = list(iter_documents(root))
    assert docs[0].text == "caf� latin-1"
    assert "bad encoding" in caplog.text
