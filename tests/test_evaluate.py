import json

import pytest

from findex.corpus import iter_documents
from findex.evaluate import main as eval_main
from findex.evaluate import precision_at_k
from findex.index import Index
from findex.sanity import main as sanity_main


@pytest.mark.parametrize(
    ("ranked", "relevant", "expected"),
    [
        (["a", "b", "c", "d", "e"], {"a", "c"}, 0.4),
        (["a", "b", "c", "d", "e", "f"], {"f"}, 0.0),  # шосте місце не рахується
        (["a"], {"a"}, 0.2),                           # порожні місця — промах
        ([], {"a"}, 0.0),
    ],
)
def test_precision_at_k(ranked, relevant, expected):
    assert precision_at_k(ranked, relevant, k=5) == pytest.approx(expected)


@pytest.fixture
def toy_index(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "short.txt").write_text("whale captain rare " + "x " * 10, encoding="utf-8")
    (data / "long.txt").write_text("rare captain " + "y " * 200, encoding="utf-8")
    (data / "moby.txt").write_text("whale " * 30 + "captain ship " * 5, encoding="utf-8")
    path = tmp_path / "idx.bin"
    Index.build(iter_documents(data), with_positions=True).save(path)
    return path


def test_evaluate_cli_prints_table(toy_index, tmp_path, monkeypatch, capsys):
    qrels = tmp_path / "q.json"
    qrels.write_text(json.dumps([{"query": "whale", "relevant": ["moby"]}]), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["evaluate", str(toy_index), str(qrels)])
    eval_main()
    out = capsys.readouterr().out
    assert "| `whale` | 1 | 0.20 | 0.20 | 0.20 |" in out


def test_sanity_cli_runs(toy_index, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["sanity", str(toy_index)])
    sanity_main()
    out = capsys.readouterr().out
    assert "## 1." in out and "## 2." in out and "## 3." in out
    assert "TF-IDF їх не розрізняє" in out
