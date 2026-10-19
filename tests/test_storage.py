import pytest

from findex.corpus import iter_documents
from findex.index import build_index
from findex.storage import load, save


@pytest.mark.parametrize("fmt", ["pickle", "json"])
@pytest.mark.parametrize("positions", [False, True])
def test_roundtrip(tmp_path, fmt, positions):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.txt").write_text("the cat sat on the mat", encoding="utf-8")
    (root / "b.txt").write_text("the dog sat", encoding="utf-8")

    postings, metas = build_index(iter_documents(root), with_positions=positions)
    out = tmp_path / "index.bin"
    save(postings, metas, out, fmt)

    assert load(out) == (postings, metas)