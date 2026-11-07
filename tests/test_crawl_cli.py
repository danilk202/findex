import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import findex.crawler
from findex.cli import app
from findex.corpus import iter_documents, read_text
from findex.crawler import CrawlConfig

runner = CliRunner()


def site_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/robots.txt":
        return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
    pages = {
        "/": ("Home", ["/whale", "/ship", "/private/x", "/gone"]),
        "/whale": ("The White Whale", ["/"]),
        "/ship": ("Ship log", ["/"]),
    }
    if path not in pages:
        return httpx.Response(404)
    title, links = pages[path]
    a = "".join(f'<a href="{h}">{h}</a>' for h in links)
    body = f"<title>{title}</title><p>{title.lower()} text, call me Ishmael</p>{a}"
    return httpx.Response(200, html=body)


@pytest.fixture
def fake_net(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI створює клієнта сам; підміняємо make_client, щоб він ходив у MockTransport."""
    real = findex.crawler.make_client

    def patched(config: CrawlConfig, transport: httpx.AsyncBaseTransport | None = None):
        return real(config, httpx.MockTransport(site_handler))

    monkeypatch.setattr(findex.crawler, "make_client", patched)


def crawl(tmp_path: Path, *extra: str):
    out, log, stats = tmp_path / "pages.jsonl", tmp_path / "crawl.log", tmp_path / "s.json"
    args = ["crawl", "http://site.test/", "-o", str(out), "--log", str(log), "--delay", "0"]
    args += ["--no-live", "--stats-json", str(stats), *extra]
    return runner.invoke(app, args), out, log, stats


def test_crawl_writes_jsonl_log_and_stats(fake_net: None, tmp_path: Path):
    r, out, log, stats = crawl(tmp_path)
    assert r.exit_code == 0, r.output
    assert "crawled 3 pages" in r.stdout
    lines = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert {x["url"] for x in lines} == {
        "http://site.test/",
        "http://site.test/whale",
        "http://site.test/ship",
    }
    assert {"url", "title", "text", "status", "depth", "bytes", "elapsed_ms"} <= set(lines[0])
    text = log.read_text("utf-8")
    assert "ROBOTS   disallowed" in text and "/private/x" in text  # кожен URL у лозі
    assert "404" in text and "ERROR" in text
    summary = json.loads(stats.read_text("utf-8"))
    assert summary["pages"] == 3 and summary["errors"] == 1 and summary["robots_blocked"] == 1


def test_crawl_max_pages(fake_net: None, tmp_path: Path):
    r, out, _, _ = crawl(tmp_path, "--max-pages", "2")
    assert r.exit_code == 0, r.output
    assert len(out.read_text("utf-8").splitlines()) == 2


def test_crawl_debug_mode_logs_no_blocking(fake_net: None, tmp_path: Path):
    r, _, log, _ = crawl(tmp_path, "--debug")
    assert r.exit_code == 0, r.output
    text = log.read_text("utf-8")
    assert "asyncio debug mode on" in text
    assert "Executing" not in text  # жодного «Executing <Task ...> took ...»


def test_crawl_bad_seed(tmp_path: Path):
    r = runner.invoke(app, ["crawl", "mailto:x@y.z", "-o", str(tmp_path / "p.jsonl"), "--no-live"])
    assert r.exit_code == 1 and "no valid http(s) seed URL" in r.stderr


def test_crawl_then_index_then_search(fake_net: None, tmp_path: Path):
    r, out, _, _ = crawl(tmp_path)
    assert r.exit_code == 0, r.output
    idx = tmp_path / "web.bin"
    r = runner.invoke(app, ["index", str(out), "-o", str(idx)])
    assert r.exit_code == 0, r.output
    assert "indexed 3 documents" in r.stdout and "jsonl" in r.stdout
    r = runner.invoke(app, ["search", str(idx), "white whale", "--json"])
    data = json.loads(r.stdout)
    top = data["results"][0]
    assert top["title"] == "The White Whale"
    assert "**" not in top["snippet"] and "whale" in top["snippet"].lower()  # сніпет із JSONL


# --- JSONL як корпус -----------------------------------------------------------------


def test_iter_documents_reads_jsonl_lazily(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    f = tmp_path / "p.jsonl"
    rows = [
        {"url": "http://a/1", "title": "One", "text": "first page"},
        {"url": "http://a/2", "title": "", "text": "second"},
    ]
    f.write_text(
        json.dumps(rows[0]) + "\n\n" + json.dumps(rows[1]) + '\n{"url": "http://a/3", "te',
        "utf-8",
    )
    docs = list(iter_documents(f))
    assert [d.doc_id for d in docs] == [0, 1]  # порожній і обірваний рядки пропущено
    assert [d.title for d in docs] == ["One", "http://a/2"]  # без title — URL
    assert docs[0].text == "One\nfirst page"
    assert str(docs[1].path).endswith("p.jsonl#3")  # локатор: номер рядка у файлі
    assert read_text(docs[1].path) == "\nsecond"
    assert "bad JSON" in caplog.text


def test_index_accepts_jsonl_only_as_file(tmp_path: Path):
    r = runner.invoke(app, ["index", str(tmp_path / "nope.jsonl")])
    assert r.exit_code == 1 and "corpus directory not found" in r.stderr
