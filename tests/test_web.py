# Веб-шар без справжнього файлу індексу: індекс збирається в пам'яті з кількох рядків,
# тексти для сніпетів — зі словника, а в застосунок він потрапляє через dependency_overrides.

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.corpus import Document
from findex.index import Index
from findex.models import DocMeta
from findex.web.app import create_app
from findex.web.deps import current_service, require_service
from findex.web.service import SearchService, segments
from findex.web.settings import Settings

TEXTS = {
    "moby": "Title: Moby Dick\nAuthor: Herman Melville\n\nCall me Ishmael. The white whale swam.\n"
    "The ship sailed after the whale.",
    "ships": "A book about ships. The ship and the sea and one more ship.",
    "java": "java coffee and more coffee",
    "loop": "python event loop. The event loop runs coroutines.",
}


def build_index(positions: bool = True) -> Index:
    docs = (Document(i, Path(f"{name}.txt"), text) for i, (name, text) in enumerate(TEXTS.items()))
    return Index.build(docs, with_positions=positions)


def texts(meta: DocMeta) -> str:
    return TEXTS[Path(meta.path).stem]


@pytest.fixture
def service() -> SearchService:
    return SearchService(build_index(), texts, snippet_width=20)


@pytest.fixture
def app() -> FastAPI:
    # Settings з неіснуючим шляхом: якщо тест випадково запустить lifespan — індексу не буде
    return create_app(Settings(index_path=Path("no-such-index.bin")))


@pytest.fixture
def client(app: FastAPI, service: SearchService) -> Iterator[TestClient]:
    app.dependency_overrides[current_service] = lambda: service
    # без `with`: lifespan не запускається, файл індексу не читається
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- /search ---------------------------------------------------------------


def test_search_ok(client: TestClient):
    r = client.get("/search", params={"q": "whale OR ship", "k": 2})
    assert r.status_code == 200
    data = r.json()
    assert data["tree"] == "(whale OR ship)"
    assert data["total"] == 2 and data["pages"] == 1
    top = data["results"][0]
    assert top["rank"] == 1 and top["title"] == "Moby Dick"  # назва з шапки Gutenberg
    assert top["score"] > data["results"][1]["score"]
    # highlights вказують саме на терми в snippet
    words = {top["snippet"][s:e].lower() for s, e in top["highlights"]}
    assert words <= {"whale", "ship"} and words
    assert "\x02" not in top["snippet"]
    assert r.headers["x-request-id"]


def test_search_pages(client: TestClient):
    first = client.get("/search", params={"q": "the", "k": 1}).json()
    second = client.get("/search", params={"q": "the", "k": 1, "page": 2}).json()
    assert first["total"] == 3 and first["pages"] == 3
    assert second["page"] == 2 and second["results"][0]["rank"] == 2
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]
    beyond = client.get("/search", params={"q": "the", "k": 1, "page": 9}).json()
    assert beyond["results"] == [] and beyond["total"] == 3


def test_search_tfidf_and_phrase(client: TestClient):
    r = client.get("/search", params={"q": '"event loop"', "scorer": "tfidf"})
    assert r.status_code == 200
    assert [x["doc_id"] for x in r.json()["results"]] == [3]


def test_search_not_found(client: TestClient):
    data = client.get("/search", params={"q": "zebra"}).json()
    assert data["total"] == 0 and data["results"] == []


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({}, "q"),  # q обов'язковий
        ({"q": "   "}, "q"),  # лише пробіли після strip
        ({"q": "x" * 501}, "q"),
        ({"q": "whale", "k": 0}, "k"),
        ({"q": "whale", "k": 51}, "k"),
        ({"q": "whale", "k": "ten"}, "k"),
        ({"q": "whale", "page": 0}, "page"),
        ({"q": "whale", "scorer": "pagerank"}, "scorer"),
        ({"q": "whale", "limit": 5}, "limit"),  # невідомий параметр — теж 422
        ({"q": "(whale"}, "q"),  # запит не розібрався
        ({"q": "whale AND"}, "q"),
    ],
)
def test_search_422(client: TestClient, params: dict[str, object], field: str):
    r = client.get("/search", params=params)  # type: ignore[arg-type]
    assert r.status_code == 422
    assert any(e["loc"][-1] == field for e in r.json()["detail"])


def test_phrase_without_positions_is_422(app: FastAPI):
    plain = SearchService(build_index(positions=False), texts)
    app.dependency_overrides[current_service] = lambda: plain
    r = TestClient(app).get("/search", params={"q": '"white whale"'})
    assert r.status_code == 422
    assert "positions" in r.json()["detail"][0]["msg"]


# --- /docs, /stats -----------------------------------------------------------


def test_document(client: TestClient):
    r = client.get("/docs/0", params={"offset": 7, "limit": 9})
    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "Moby Dick" and data["chars"] == len(TEXTS["moby"])
    assert data["title"] == "Moby Dick"


@pytest.mark.parametrize("doc_id", [4, 999])
def test_document_404(client: TestClient, doc_id: int):
    r = client.get(f"/docs/{doc_id}")
    assert r.status_code == 404
    assert r.json() == {"detail": f"document {doc_id} not found"}


@pytest.mark.parametrize("path", ["/docs/-1", "/docs/abc", "/docs/0?limit=0"])
def test_document_422(client: TestClient, path: str):
    assert client.get(path).status_code == 422


def test_stats(client: TestClient):
    data = client.get("/stats").json()
    assert data["documents"] == 4 and data["positions"] is True
    assert data["top_terms"][0]["term"] == "the"


# --- /health і стан без індексу ----------------------------------------------


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "documents": 4}


def test_health_503_when_index_not_loaded(app: FastAPI):
    # без override і без lifespan у застосунку індексу немає
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "loading"}
    s = c.get("/search", params={"q": "whale"})
    assert s.status_code == 503 and s.headers["retry-after"] == "5"
    assert c.get("/stats").status_code == 503
    page = c.get("/")
    assert page.status_code == 503 and "завантажується" in page.text


def wait_until_loaded(c: TestClient):
    for _ in range(500):
        r = c.get("/health")
        if r.json()["status"] != "loading":
            return r
        time.sleep(0.01)
    raise AssertionError("index is still loading")


def test_lifespan_loads_index(tmp_path: Path):
    path = tmp_path / "index.bin"
    build_index().save(path)
    with TestClient(create_app(Settings(index_path=path))) as c:
        wait_until_loaded(c)  # індекс вантажиться у фоновій задачі
        assert c.get("/health").json() == {"status": "ok", "documents": 4}
        assert c.get("/stats").json()["documents"] == 4


def test_lifespan_bad_index_file(tmp_path: Path):
    path = tmp_path / "index.bin"
    path.write_text("not an index")
    with TestClient(create_app(Settings(index_path=path))) as c:
        r = wait_until_loaded(c)
        assert r.status_code == 503
        assert r.json()["status"] == "error" and "not a findex index" in r.json()["detail"]


# --- 500: стек у лог, назовні — ні ------------------------------------------


def test_unexpected_error_is_500_without_stack(
    app: FastAPI, service: SearchService, caplog: pytest.LogCaptureFixture
):
    def broken() -> SearchService:
        raise RuntimeError("secret internal detail")

    app.dependency_overrides[require_service] = broken
    c = TestClient(app)
    with caplog.at_level(logging.ERROR, logger="findex.web"):
        r = c.get("/search", params={"q": "whale"}, headers={"X-Request-ID": "req-42"})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error", "request_id": "req-42"}
    assert "secret" not in r.text and "Traceback" not in r.text
    assert r.headers["x-request-id"] == "req-42"
    rec = next(r for r in caplog.records if r.name == "findex.web")
    assert rec.exc_info is not None and "GET /search" in rec.getMessage()
    assert rec.__dict__["request_id"] == "req-42"  # той самий id, що й у відповіді


def test_request_id_generated_and_bad_one_replaced(client: TestClient):
    r = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    rid = r.headers["x-request-id"]
    assert rid != "bad id with spaces" and len(rid) == 12


# --- HTML ----------------------------------------------------------------------


def test_home_page(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "Пошук по 25 класичних книжках" in r.text
    assert "/static/htmx.min.js" in r.text


def test_results_page_highlights_and_escapes(client: TestClient):
    r = client.get("/", params={"q": "white whale"})
    assert r.status_code == 200
    assert "<mark>whale</mark>" in r.text.lower()
    assert "Moby Dick" in r.text and "Herman Melville" in r.text
    evil = client.get("/", params={"q": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in evil.text
    assert "&lt;script&gt;" in evil.text


def test_results_partial_for_htmx(client: TestClient):
    r = client.get("/", params={"q": "the", "k": 1}, headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert "<html" not in r.text  # лише блок результатів
    assert 'class="pager"' in r.text and "page=2" in r.text
    assert r.headers["vary"] == "HX-Request"
    restore = client.get(
        "/",
        params={"q": "the"},
        headers={"HX-Request": "true", "HX-History-Restore-Request": "true"},
    )
    assert "<html" in restore.text


def test_results_page_errors(client: TestClient):
    bad = client.get("/", params={"q": "(whale"})
    assert bad.status_code == 422 and "Не вдалося розібрати запит" in bad.text
    bad_k = client.get("/", params={"q": "whale", "k": "0"})
    assert bad_k.status_code == 422 and "Некоректні параметри" in bad_k.text
    empty = client.get("/", params={"q": "zebra"})
    assert "Нічого не знайдено" in empty.text


def test_view_document(client: TestClient):
    r = client.get("/view/0", params={"q": "whale"})
    assert r.status_code == 200
    assert '<mark id="hit">whale</mark>' in r.text
    assert "Herman Melville" in r.text
    assert client.get("/view/0", params={"q": "(bad"}).status_code == 200  # без підсвітки
    missing = client.get("/view/77")
    assert missing.status_code == 404 and "немає в індексі" in missing.text


def test_openapi_moved_from_docs(client: TestClient):
    assert client.get("/api/docs").status_code == 200
    paths = client.get("/api/openapi.json").json()["paths"]
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= set(paths)
    assert "/" not in paths  # HTML-сторінки не в схемі API


# --- service без HTTP ----------------------------------------------------------


def test_segments():
    assert segments("a \x02whale\x03 b") == (("a ", False), ("whale", True), (" b", False))
    assert segments("\x02x\x03") == (("x", True),)
    assert segments("plain") == (("plain", False),)


def test_document_chunks_and_focus():
    long_text = "".join(f"line {i} of filler text\n" for i in range(3000)) + "the whale\n"
    index = Index.build([Document(0, Path("big.txt"), long_text)])
    svc = SearchService(index, lambda _: long_text)
    first = svc.document(0)
    assert first is not None and first.part == 1 and first.parts > 1 and first.focus is None
    hit = svc.document(0, "whale")
    assert hit is not None and hit.part == hit.parts  # кит у останньому шматку
    assert hit.focus is not None and hit.text[hit.focus] == ("whale", True)
    clamp = svc.document(0, None, part=10_000)
    assert clamp is not None and clamp.part == clamp.parts
    assert svc.document(5) is None


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "x.bin"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.index_path == tmp_path / "x.bin" and s.log_level == "DEBUG"
    monkeypatch.setenv("SNIPPET_WIDTH", "5")
    with pytest.raises(ValueError):
        Settings()


# --- findex serve ----------------------------------------------------------------


def test_serve_starts_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "index.bin"
    build_index().save(path)
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("uvicorn.run", lambda target, **kw: calls.append((target, kw)))
    monkeypatch.delenv("INDEX_PATH", raising=False)
    r = CliRunner().invoke(cli_app, ["serve", "--index", str(path), "--port", "9001", "-w", "3"])
    assert r.exit_code == 0, r.output
    target, kw = calls[0]
    assert target == "findex.web.app:create_app" and kw["factory"] is True
    assert kw["port"] == 9001 and kw["workers"] == 3 and kw["host"] == "127.0.0.1"
    import os

    assert os.environ["INDEX_PATH"] == str(path)  # воркери прочитають його в Settings


def test_serve_reads_port_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "index.bin"
    build_index().save(path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("uvicorn.run", lambda target, **kw: calls.append(kw))
    monkeypatch.setenv("INDEX_PATH", str(path))
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("HOST", "0.0.0.0")
    r = CliRunner().invoke(cli_app, ["serve"])
    assert r.exit_code == 0, r.output
    assert calls[0]["port"] == 10000 and calls[0]["host"] == "0.0.0.0"


def test_serve_missing_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: pytest.fail("must not start"))
    r = CliRunner().invoke(cli_app, ["serve", "--index", str(tmp_path / "nope.bin")])
    assert r.exit_code == 1 and "index file not found" in r.stderr


# --- логи ------------------------------------------------------------------------


def test_log_config_applies_and_filters():
    import logging.config

    from findex.web.logs import QuietHealthFilter, RequestIdFilter, log_config, request_id

    logging.config.dictConfig(log_config("DEBUG"))  # словник валідний для dictConfig
    rec = logging.LogRecord("x", logging.INFO, "", 0, "m", None, None)
    token = request_id.set("abc")
    assert RequestIdFilter().filter(rec) and rec.__dict__["request_id"] == "abc"
    request_id.reset(token)

    def access(path: str, status: int) -> logging.LogRecord:
        args = ("1.2.3.4:5", "GET", path, "1.1", status)
        return logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "%s", args, None)

    quiet = QuietHealthFilter()
    assert not quiet.filter(access("/health", 200))
    assert quiet.filter(access("/health", 503))  # невдала перевірка — у лог
    assert quiet.filter(access("/search?q=x", 200))
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": False})
    for name in ("findex", "uvicorn", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.NOTSET)


# --- корпус для Docker ------------------------------------------------------------


def test_fetch_corpus_skips_existing_and_retries(tmp_path: Path):
    import httpx

    from findex.gutenberg import BOOKS, fetch_corpus

    seen: list[str] = []
    failed: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path not in failed:  # перша спроба кожної книжки — 503
            failed.add(request.url.path)
            return httpx.Response(503)
        return httpx.Response(200, content=b"Title: X\n")

    (tmp_path / "moby.txt").write_text("already here")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    written = fetch_corpus(tmp_path, mirror="https://m.test/", client=client, pause=0)
    assert len(written) == len(BOOKS) - 1
    assert "/cache/epub/2701/pg2701.txt" not in seen  # moby уже є
    assert "/cache/epub/84/pg84.txt" in seen
    assert len(seen) == 2 * (len(BOOKS) - 1)
    assert (tmp_path / "moby.txt").read_text() == "already here"
