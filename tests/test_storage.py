from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import build_index
from findex.storage import IndexFormatError, load, save


@pytest.mark.parametrize("fmt", ["pickle", "json"])
@pytest.mark.parametrize("positions", [False, True])
def test_roundtrip(small_corpus: Path, tmp_path: Path, fmt, positions):
    postings, metas = build_index(iter_documents(small_corpus), with_positions=positions)
    out = tmp_path / "index.bin"
    save(postings, metas, out, fmt)
    assert load(out) == (postings, metas)


@pytest.mark.parametrize("content", [b"", b"not an index", b"{broken json", b"\x80\x05N."])
def test_not_an_index(tmp_path: Path, content: bytes):
    path = tmp_path / "bad.bin"
    path.write_bytes(content)
    with pytest.raises(IndexFormatError):
        load(path)


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.bin")
