"""Спільні фікстури.

Scope: function (типово) — свіжа копія на кожен тест, для всього, що тест може змінити
або закрити. session — один раз на весь прогін, для дорогого й незмінного (збірка
індексу для тестів CLI).
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index

type MakeCorpus = Callable[[dict[str, str]], Path]

SMALL = {"a": "the cat sat on the mat", "b": "the dog sat"}

SEARCH = {
    "d0": "python async await event loop",
    "d1": "python java loop event",
    "d2": "java coffee",
    "d3": "Python event loop. The event loop runs coroutines with async and await.",
}


def write_corpus(root: Path, texts: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (root / f"{name}.txt").write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def make_corpus(tmp_path: Path) -> MakeCorpus:
    """Фабрика: make_corpus({"name": "text", ...}) -> каталог з .txt."""
    counter = iter(range(1_000))
    return lambda texts: write_corpus(tmp_path / f"corpus{next(counter)}", texts)


@pytest.fixture
def small_corpus(make_corpus: MakeCorpus) -> Path:
    return make_corpus(SMALL)


@pytest.fixture
def small_index(small_corpus: Path) -> Index:
    """Індекс без позицій: a = 'the cat sat on the mat' (6), b = 'the dog sat' (3)."""
    return Index.build(iter_documents(small_corpus))


@pytest.fixture(scope="session")
def search_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_corpus(tmp_path_factory.mktemp("search") / "data", SEARCH)


@pytest.fixture(scope="session")
def search_index_file(search_corpus: Path) -> Path:
    """Збирається один раз на сесію; тести лише читають файл."""
    path = search_corpus.parent / "index.bin"
    Index.build(iter_documents(search_corpus), with_positions=True).save(path)
    return path
