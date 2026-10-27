from collections.abc import Mapping

import pytest

from findex.corpus import iter_documents
from findex.index import Index, open_index


@pytest.fixture
def index(tmp_path):
    (tmp_path / "a.txt").write_text("the cat sat on the mat", encoding="utf-8")
    (tmp_path / "b.txt").write_text("the dog sat", encoding="utf-8")
    return Index.build(iter_documents(tmp_path))


def test_mapping_protocol(index):
    assert isinstance(index, Mapping)
    assert len(index) == 6  # the cat sat on mat dog
    assert "cat" in index and "zebra" not in index
    assert [p.doc_id for p in index["sat"]] == [0, 1]
    assert set(index) == {"the", "cat", "sat", "on", "mat", "dog"}
    with pytest.raises(KeyError):
        index["zebra"]


def test_mapping_mixins_for_free(index):
    # get / keys / items нікуди не писали — їх дав Mapping
    assert index.get("zebra") is None
    assert len(index.keys()) == 6
    assert dict(index.items())["dog"][0].doc_id == 1


def test_stats(index):
    assert index.num_docs == 2
    assert index.avg_doc_length == 4.5  # (6 + 3) / 2
    assert index.df("the") == 2
    assert index.df("dog") == 1
    assert index.df("zebra") == 0


def test_avg_doc_length_is_cached(index):
    assert "avg_doc_length" not in index.__dict__
    _ = index.avg_doc_length
    assert index.__dict__["avg_doc_length"] == 4.5


def test_repr(index):
    assert repr(index) == "Index(docs=2, terms=6, avg_len=4.5, positions=False)"


def test_open_index_closes_on_exception(tmp_path, index):
    path = tmp_path / "idx.bin"
    index.save(path)
    with pytest.raises(RuntimeError), open_index(path) as idx:
        assert idx.num_docs == 2
        raise RuntimeError("boom")
    assert idx.closed
    with pytest.raises(ValueError):
        idx["the"]


@pytest.mark.parametrize("fmt", ["pickle", "json"])
def test_save_load_roundtrip(tmp_path, index, fmt):
    path = tmp_path / f"idx.{fmt}"
    index.save(path, fmt)
    with open_index(path) as idx:
        assert idx == index  # __eq__ теж від Mapping: порівнює як словники
