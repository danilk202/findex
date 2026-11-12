# Лаба 8: pytest-benchmark. У звичайному прогоні (--benchmark-disable в pyproject) кожна
# функція виконується один раз, як smoke-тест. У CI окрема задача вмикає заміри і
# порівнює з попереднім прогоном на main (див. .github/workflows/ci.yml).

import random
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import Document
from findex.engine import Searcher
from findex.index import Index, build_index
from findex.profbench import Lab3Index
from findex.scoring import BM25
from findex.snippets import find_hits

QUERY = "w1 w7 w42"  # частий, середній і рідший терм: Zipf робить w1 найчастішим


def synthetic(n_docs: int = 2000, words: int = 300, vocab: int = 5000) -> list[Document]:
    # Детермінований корпус без мережі: частоти слів за законом Ціпфа, як у живій мові.
    rng = random.Random(8)
    names = [f"w{i}" for i in range(1, vocab + 1)]
    weights = [1 / i for i in range(1, vocab + 1)]
    return [
        Document(i, Path(f"d{i}.txt"), " ".join(rng.choices(names, weights, k=words)))
        for i in range(n_docs)
    ]


@pytest.fixture(scope="module")
def corpus() -> list[Document]:
    return synthetic()


@pytest.fixture(scope="module")
def index(corpus: list[Document]) -> Index:
    return Index(*build_index(iter(corpus), with_positions=True))


@pytest.mark.benchmark(group="index")
def test_build_index(benchmark: BenchmarkFixture, corpus: list[Document]):
    postings, metas = benchmark(lambda: build_index(iter(corpus), with_positions=True))
    assert len(metas) == len(corpus) and "w1" in postings


@pytest.mark.benchmark(group="index")
def test_arrays_from_postings(benchmark: BenchmarkFixture, corpus: list[Document]):
    postings, metas = build_index(iter(corpus), with_positions=True)
    index = benchmark(Index, postings, metas)
    assert index.num_postings == sum(map(len, postings.values()))


@pytest.mark.benchmark(group="index")
def test_load(benchmark: BenchmarkFixture, index: Index, tmp_path: Path):
    path = tmp_path / "index.bin"
    index.save(path)
    loaded = benchmark(Index.load, path)
    assert loaded.num_docs == index.num_docs


@pytest.mark.benchmark(group="search")
@pytest.mark.parametrize("backend", ["numpy", "python"])
def test_search(benchmark: BenchmarkFixture, index: Index, backend: str):
    # python — скорер з лаби 3 на його рідних списках Posting (Lab3Index)
    if backend == "numpy":
        searcher = Searcher(index, BM25())
    else:
        searcher = Searcher(Lab3Index(index), BM25(), backend="python")
    query = QUERY.replace(" ", " OR ")
    searcher.search(query)  # дерево запиту потрапляє в lru_cache — далі міряється рейтинг
    results = benchmark(searcher.search, query)
    assert len(results) == 10


@pytest.mark.benchmark(group="search")
def test_snippet_hits(benchmark: BenchmarkFixture, corpus: list[Document]):
    text = "\n".join(d.text for d in corpus[:200])
    hits = benchmark(find_hits, text, {"w7", "w42"})
    assert hits
