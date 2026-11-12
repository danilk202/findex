# Лаба 8: семантичний пошук без справжньої моделі. FakeEmbedder кладе синоніми в один
# вимір, тож «phantom» близький до «ghost», хоча спільних слів у них немає.

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.corpus import Document
from findex.index import Index
from findex.semantic import (
    DIM,
    SemanticIndex,
    SemanticSearch,
    SemanticUnavailableError,
    build_embeddings,
    corpus_fingerprint,
    fragments,
    query_text,
    rrf,
)
from findex.web.app import create_app
from findex.web.deps import current_service
from findex.web.service import SearchService
from findex.web.settings import Settings

CONCEPTS = {
    "sea": 0, "ocean": 0, "whale": 0, "ship": 0, "sailor": 0, "harpoon": 0,
    "ghost": 1, "spirit": 1, "phantom": 1, "haunted": 1,
    "love": 2, "marriage": 2, "romance": 2, "wedding": 2,
    "java": 3, "coffee": 3, "espresso": 3,
}  # fmt: skip

TEXTS = [
    "Call me Ishmael. The white whale swam in the sea and the ship followed.",
    "A haunted house. The ghost of Christmas past was a spirit.",
    "Pride and prejudice: love and marriage in the country.",
    "java coffee and more coffee",
]


class FakeEmbedder:
    calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        FakeEmbedder.calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, 10] = 0.1  # «шум», спільний для всіх
            for w in text.lower().replace(".", " ").replace(",", " ").split():
                if w in CONCEPTS:
                    out[row, CONCEPTS[w]] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def make_index() -> Index:
    return Index.build((Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(TEXTS)))


@pytest.fixture
def index() -> Index:
    return make_index()


@pytest.fixture
def emb(index: Index, tmp_path: Path) -> Path:
    out = tmp_path / "emb"
    build_embeddings(index, lambda d: TEXTS[d], FakeEmbedder(), out, words=6, batch=3)
    return out


@pytest.fixture
def semantic(index: Index, emb: Path) -> SemanticSearch:
    return SemanticSearch(SemanticIndex.load(emb, index), FakeEmbedder())


# --- фрагменти і файли -------------------------------------------------------


def test_fragments_cover_all_words():
    text = "  one two three\nfour five six seven  "
    spans = fragments(text, words=3)
    assert [text[s:e] for s, e in spans] == ["one two three", "four five six", "seven"]
    assert fragments("", 3) == []


def test_query_text_drops_operators():
    assert query_text('(ghost OR spirit) AND NOT "white whale"') == "ghost spirit white whale"


def test_build_and_load(index: Index, emb: Path):
    meta = json.loads((emb / "meta.json").read_text())
    vectors = np.load(emb / "vectors.npy")
    frags = np.load(emb / "fragments.npy")
    assert vectors.dtype == np.float16 and vectors.shape == (meta["fragments"], DIM)
    assert frags[:, 0].tolist() == sorted(frags[:, 0].tolist())  # за doc_id
    assert np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1, atol=1e-3)
    assert meta["corpus"] == corpus_fingerprint(index)
    loaded = SemanticIndex.load(emb, index)
    assert loaded.present.tolist() == [0, 1, 2, 3]


def test_other_corpus_is_rejected(emb: Path):
    other = Index.build([Document(0, Path("x.txt"), "something else entirely")])
    with pytest.raises(SemanticUnavailableError):
        SemanticIndex.load(emb, other)


def test_open_without_embeddings(index: Index, tmp_path: Path):
    with pytest.raises(SemanticUnavailableError):
        SemanticSearch.open(tmp_path / "nothing", index)


# --- рейтинг ------------------------------------------------------------------


def test_paraphrase_without_shared_words(semantic: SemanticSearch):
    hits = semantic.search("phantom", k=2)
    assert hits[0].doc_id == 1 and hits[0].score > 0.9
    hits = semantic.search("an ocean sailor with a harpoon", k=4)
    assert hits[0].doc_id == 0


def test_doc_scores_take_best_fragment(semantic: SemanticSearch):
    q = semantic.encode_query("whale")
    scores = semantic.index.doc_scores(q)
    best, s, e = semantic.index.best_fragment(q, 0)
    assert scores[0] == pytest.approx(best) and "whale" in TEXTS[0][s:e]


def test_query_vector_is_cached(semantic: SemanticSearch):
    before = FakeEmbedder.calls
    semantic.encode_query("ghost OR spirit")
    semantic.encode_query("ghost OR spirit")
    assert FakeEmbedder.calls == before + 1


def test_rrf():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert [d for d, _ in fused] == [1, 3, 2, 4]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


# --- сервіс, API, сторінка, CLI ----------------------------------------------


def test_service_modes(index: Index, semantic: SemanticSearch):
    svc = SearchService(index, lambda m: TEXTS[int(Path(m.path).stem[1:])], semantic=semantic)
    lexical = svc.search("phantom")
    assert lexical.total == 0
    sem = svc.search("phantom", mode="semantic")
    assert sem.hits[0].doc_id == 1 and sem.mode == "semantic" and sem.total == 4
    hybrid = svc.search("ghost OR coffee", mode="hybrid")
    assert {h.doc_id for h in hybrid.hits[:2]} == {1, 3}
    # слово є в документі -> сніпет з підсвіченим словом
    assert any(hit for _, hit in hybrid.hits[0].snippet)


def test_service_without_semantic(index: Index):
    svc = SearchService(index, lambda m: TEXTS[0])
    with pytest.raises(SemanticUnavailableError):
        svc.search("ghost", mode="semantic")


@pytest.fixture
def client(index: Index, semantic: SemanticSearch):
    app = create_app(Settings(index_path=Path("no-such-index.bin")))
    svc = SearchService(index, lambda m: TEXTS[int(Path(m.path).stem[1:])], semantic=semantic)
    app.dependency_overrides[current_service] = lambda: svc
    yield TestClient(app)


def test_api_semantic(client: TestClient):
    r = client.get("/search", params={"q": "phantom", "mode": "semantic", "k": 2})
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "semantic" and data["results"][0]["doc_id"] == 1
    assert client.get("/search", params={"q": "x", "mode": "vibes"}).status_code == 422


def test_api_semantic_unavailable(index: Index):
    app = create_app(Settings(index_path=Path("no-such-index.bin")))
    app.dependency_overrides[current_service] = lambda: SearchService(index, lambda m: "")
    r = TestClient(app).get("/search", params={"q": "ghost", "mode": "semantic"})
    assert r.status_code == 503
    page = TestClient(app).get("/", params={"q": "ghost", "mode": "semantic"})
    assert page.status_code == 503 and "Семантичний" in page.text
    assert 'name="mode"' not in TestClient(app).get("/").text  # без ембедингів — без вибору


def test_page_semantic(client: TestClient):
    r = client.get("/", params={"q": "phantom", "mode": "semantic"})
    assert r.status_code == 200 and "косинус" in r.text and 'name="mode"' in r.text


def test_cli_semantic_and_embed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus = tmp_path / "data"
    corpus.mkdir()
    for i, t in enumerate(TEXTS):
        (corpus / f"d{i}.txt").write_text(t, encoding="utf-8")
    index_file = tmp_path / "index.bin"
    runner = CliRunner()
    assert runner.invoke(cli_app, ["index", str(corpus), "-o", str(index_file)]).exit_code == 0
    monkeypatch.setattr("findex.semantic.ensure_model", lambda d=None: tmp_path)
    monkeypatch.setattr("findex.semantic.MiniLM", lambda d, threads=None: FakeEmbedder())
    emb = tmp_path / "emb"
    r = runner.invoke(cli_app, ["embed", str(index_file), "-o", str(emb), "--words", "10"])
    assert r.exit_code == 0, r.output
    assert "embedded 6 fragments of 4 documents" in r.output
    r = runner.invoke(cli_app, ["embed", str(index_file), "-o", str(emb), "--skip-if-fresh"])
    assert r.exit_code == 0 and "up to date" in r.output
    r = runner.invoke(
        cli_app,
        ["search", str(index_file), "phantom", "--mode", "semantic", "--emb", str(emb), "--json"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["mode"] == "semantic" and data["results"][0]["title"] == "d1"
    r = runner.invoke(cli_app, ["search", str(index_file), "x", "-m", "hybrid", "--emb", "nope"])
    assert r.exit_code == 1 and "no embeddings" in r.output


# --- завантаження моделі, справжня модель (лише локально), P@5 -------------------


class FakeStream:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, size: int):
        yield self.body


def test_ensure_model_checks_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import hashlib

    import httpx

    import findex.semantic as sem

    good = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setattr(sem, "MODEL_FILES", {"onnx/model.onnx": good})
    calls: list[str] = []

    def stream(method: str, url: str, **kw: object) -> FakeStream:
        calls.append(url)
        return FakeStream(b"weights")

    monkeypatch.setattr(httpx, "stream", stream)
    assert sem.ensure_model(tmp_path) == tmp_path
    assert (tmp_path / "onnx/model.onnx").read_bytes() == b"weights"
    sem.ensure_model(tmp_path)  # уже є і sha збігається — не качає вдруге
    assert len(calls) == 1 and sem.MODEL_REVISION in calls[0]

    monkeypatch.setattr(sem, "MODEL_FILES", {"tokenizer.json": "0" * 64})
    with pytest.raises(SemanticUnavailableError, match="sha256"):
        sem.ensure_model(tmp_path)
    assert not (tmp_path / "tokenizer.json").exists()


def test_model_missing(tmp_path: Path):
    from findex.semantic import MiniLM

    with pytest.raises(SemanticUnavailableError):
        MiniLM(tmp_path)


@pytest.mark.skipif(
    not (Path.home() / ".cache/findex/all-MiniLM-L6-v2/onnx/model.onnx").is_file(),
    reason="справжня модель — лише якщо вже скачана (findex embed)",
)
def test_real_minilm_paraphrase():
    from findex.semantic import MiniLM, default_model_dir

    m = MiniLM(default_model_dir())
    v = m.encode(["A man is eating food.", "A man is eating a piece of bread.", "A cheetah runs."])
    assert v.shape == (3, DIM)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > 0.6 > v[0] @ v[2]


def test_evaluate_modes(index: Index, emb: Path, tmp_path: Path, monkeypatch, capsys):
    from findex.evaluate import main as eval_main

    path = tmp_path / "index.bin"
    index.save(path)
    monkeypatch.chdir(tmp_path)  # шляхи в індексі відносні: d0.txt ... — для сніпетів
    for i, text in enumerate(TEXTS):
        (tmp_path / f"d{i}.txt").write_text(text, encoding="utf-8")
    qrels = tmp_path / "paraphrases.json"
    qrels.write_text(json.dumps([{"query": "phantom", "relevant": ["d1"]}]), encoding="utf-8")
    monkeypatch.setattr(
        "findex.semantic.SemanticSearch.open",
        lambda e, idx: SemanticSearch(SemanticIndex.load(emb, idx), FakeEmbedder()),
    )
    monkeypatch.setattr("sys.argv", ["evaluate", str(path), str(qrels), "--emb", str(emb)])
    eval_main()
    out = capsys.readouterr().out
    assert "| `phantom` | paraphrases | 1 | 0.00 | 0.20 | 0.20 | 0.20 |" in out
