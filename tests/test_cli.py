import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def run(*args: str | Path):
    return runner.invoke(app, [str(a) for a in args])


# --- index ---------------------------------------------------------------


def test_index_builds_file(search_corpus: Path, tmp_path: Path):
    out = tmp_path / "idx.bin"
    r = run("index", search_corpus, "--out", out)
    assert r.exit_code == 0, r.output
    assert out.is_file()
    assert "indexed 4 documents" in r.stdout
    assert "indexing" in r.stderr  # смуга прогресу — у stderr, не в stdout


def test_index_json_format(search_corpus: Path, tmp_path: Path):
    out = tmp_path / "idx.json"
    assert run("index", search_corpus, "-o", out, "--format", "json").exit_code == 0
    assert out.read_bytes()[:1] == b"{"


def test_index_missing_corpus(tmp_path: Path):
    r = run("index", tmp_path / "nope")
    assert r.exit_code == 1
    assert r.stderr.strip() == f"error: corpus directory not found: {tmp_path / 'nope'}"


def test_index_empty_corpus(tmp_path: Path):
    r = run("index", tmp_path)
    assert r.exit_code == 1
    assert "no .txt files" in r.stderr


# --- search --------------------------------------------------------------


def test_search_table(search_index_file: Path):
    r = run("search", search_index_file, 'python AND (async OR await) NOT java "event loop"')
    assert r.exit_code == 0, r.output
    assert "d0" in r.stdout and "d3" in r.stdout
    assert "d1" not in r.stdout  # java
    assert r.stderr == ""  # без -v діагностики немає


def test_search_json(search_index_file: Path):
    r = run("search", search_index_file, "python OR coffee", "--json")
    assert r.exit_code == 0
    data = json.loads(r.stdout)  # stdout — чистий JSON
    assert data["tree"] == "(python OR coffee)"
    assert {row["title"] for row in data["results"]} == {"d0", "d1", "d2", "d3"}
    scores = [row["score"] for row in data["results"]]
    assert scores == sorted(scores, reverse=True)
    assert all("\x02" not in (row["snippet"] or "") for row in data["results"])


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_search_limit_and_scorer(search_index_file: Path, scorer: str):
    r = run("search", search_index_file, "python", "-k", "1", "--scorer", scorer, "--json")
    data = json.loads(r.stdout)
    assert data["scorer"] == scorer
    assert len(data["results"]) == 1


def test_search_nothing_found(search_index_file: Path):
    r = run("search", search_index_file, "zebra")
    assert r.exit_code == 0
    assert "nothing found" in r.stdout


def test_search_missing_index(tmp_path: Path):
    r = run("search", tmp_path / "nope.bin", "whale")
    assert r.exit_code == 1
    lines = r.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: index file not found")
    assert "Traceback" not in r.output


@pytest.mark.parametrize("query", ["(whale", "whale AND", "OR", '"unclosed', "()"])
def test_search_bad_query(search_index_file: Path, query: str):
    r = run("search", search_index_file, query)
    assert r.exit_code == 2
    assert r.stderr.startswith("error: bad query:")
    assert len(r.stderr.strip().splitlines()) == 1


def test_search_not_an_index(search_corpus: Path):
    r = run("search", search_corpus / "d0.txt", "python")
    assert r.exit_code == 1
    assert "is not a findex index" in r.stderr


def test_phrase_without_positions(search_corpus: Path, tmp_path: Path):
    out = tmp_path / "np.bin"
    run("index", search_corpus, "-o", out, "--no-positions")
    r = run("search", out, '"event loop"')
    assert r.exit_code == 2
    assert "positions" in r.stderr


# --- stats ---------------------------------------------------------------


def test_stats_table(search_index_file: Path):
    r = run("stats", search_index_file, "--top", "3")
    assert r.exit_code == 0
    assert "documents" in r.stdout and "top 3 terms" in r.stdout


def test_stats_json(search_index_file: Path):
    data = json.loads(run("stats", search_index_file, "--json", "-n", "2").stdout)
    assert data["documents"] == 4
    assert data["positions"] is True
    assert len(data["top_terms"]) == 2


# --- логування і загальні опції ------------------------------------------


@pytest.mark.parametrize(
    ("flags", "info", "debug"),
    [((), False, False), (("-v",), True, False), (("-vv",), True, True)],
)
def test_verbosity(search_index_file: Path, flags: tuple[str, ...], info: bool, debug: bool):
    r = run(*flags, "search", search_index_file, "python", "--json")
    assert ("Searcher.search took" in r.stderr) is info
    assert ("candidate documents" in r.stderr) is debug
    json.loads(r.stdout)  # логи не забруднюють stdout


def test_help_and_version():
    assert run("--help").exit_code == 0
    r = run("--version")
    assert r.exit_code == 0 and r.stdout.startswith("findex ")


def test_bad_option_is_usage_error(search_index_file: Path):
    assert run("search", search_index_file, "x", "--scorer", "nope").exit_code == 2


def test_python_dash_m(search_index_file: Path):
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-m", "findex", "stats", str(search_index_file), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["documents"] == 4


def test_index_out_dir_missing(search_corpus: Path, tmp_path: Path):
    r = run("index", search_corpus, "-o", tmp_path / "no" / "dir" / "i.bin")
    assert r.exit_code == 1
    assert r.stderr == f"error: output directory not found: {tmp_path / 'no' / 'dir'}\n"


def test_index_out_is_directory(search_corpus: Path, tmp_path: Path):
    r = run("index", search_corpus, "-o", tmp_path)
    assert r.exit_code == 1
    assert r.stderr.startswith("error: --out points to a directory")


def test_unexpected_os_error_is_one_line(search_corpus: Path, tmp_path: Path, monkeypatch):
    def deny(*_args: object, **_kw: object) -> None:
        raise PermissionError(13, "Permission denied", str(tmp_path / "i.bin"))

    monkeypatch.setattr("findex.index.Index.save", deny)
    r = run("index", search_corpus, "-o", tmp_path / "i.bin")
    assert r.exit_code == 1
    assert (
        r.stderr.splitlines()[-1] == f"error: cannot access {tmp_path / 'i.bin'}: Permission denied"
    )


def test_stats_missing_index(tmp_path: Path):
    r = run("stats", tmp_path / "nope.bin")
    assert r.exit_code == 1
    assert r.stderr.startswith("error: index file not found")
