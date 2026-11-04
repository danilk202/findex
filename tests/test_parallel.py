import os
import pickle
import random
from pathlib import Path

import pytest
from conftest import MakeCorpus, write_corpus
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import corpus_files, iter_documents
from findex.index import Index, build_index
from findex.models import DocMeta, Posting
from findex.parallel import (
    EXECUTORS,
    Chunk,
    ExecutorKind,
    Partial,
    WorkerError,
    build_parallel,
    build_partial,
    cpu_count,
    gil_enabled,
    make_pool,
    map_partials,
    merge,
    split,
)

BOOKS = {
    "b00": "Call me Ishmael. Some years ago — never mind how long precisely",
    "b01": "It is a truth universally acknowledged, that a single man",
    "b02": "the whale the whale the white whale " * 20,
    "b03": "",
    "b04": "Ship and sea; sea and ship. Captain Ahab!",
    "b05": "don't stop — well-known п'ять Ґанок ß",
    "b06": "a " * 300,
    "b07": "the end",
    "b08": "whale",
}


@pytest.fixture(scope="session")
def books(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """session: тести корпус лише читають (і Hypothesis не скаржиться на function-фікстуру)."""
    return write_corpus(tmp_path_factory.mktemp("books") / "data", BOOKS)


# --- послідовний варіант == лаба 4 (окремий тест, як вимагає завдання) -------------


@pytest.mark.parametrize("positions", [False, True])
def test_merge_of_single_partial_equals_lab4(books: Path, positions: bool):
    paths = corpus_files(books)
    expected = build_index(iter_documents(books), with_positions=positions)
    got = merge([build_partial(Chunk(0, tuple(map(str, paths))), positions)])
    assert got == expected


# --- будь-яке число шматків і будь-який виконавець == лаба 4 ------------------------


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("workers", [1, 2, 4, 20])
def test_build_parallel_equals_lab4(books: Path, executor: ExecutorKind, workers: int):
    expected = build_index(iter_documents(books), with_positions=True)
    postings, metas, report = build_parallel(
        corpus_files(books), workers=workers, executor=executor, with_positions=True
    )
    assert (postings, metas) == expected
    assert report.workers == report.chunks == min(workers, len(BOOKS))
    assert Index(postings, metas) == Index(*expected)


@settings(max_examples=60, deadline=None)
@given(parts=st.integers(1, 12), shuffle=st.randoms(use_true_random=False))
def test_merge_order_does_not_matter(books: Path, parts: int, shuffle: random.Random):
    chunks = split(corpus_files(books), parts)
    partials = [build_partial(c, True) for c in chunks]
    shuffle.shuffle(partials)  # процеси завершуються в довільному порядку
    assert merge(partials) == build_index(iter_documents(books), with_positions=True)


# --- split ------------------------------------------------------------------------


@settings(max_examples=200)
@given(n=st.integers(0, 40), parts=st.integers(1, 50))
def test_split_covers_everything_in_order(n: int, parts: int):
    paths = [f"f{i:03}.txt" for i in range(n)]  # неіснуючі: розмір вважається 1
    chunks = split(paths, parts)
    assert [p for c in chunks for p in c.paths] == paths  # нічого не загубили, порядок той самий
    assert all(c.paths for c in chunks)  # жодного порожнього шматка
    assert len(chunks) == min(n, parts)  # рівні розміри -> рівно стільки шматків
    starts = [c.start for c in chunks]
    assert starts == [sum(len(c.paths) for c in chunks[:i]) for i in range(len(chunks))]


def test_split_balances_by_bytes(make_corpus: MakeCorpus):
    root = make_corpus({"a": "x" * 1000, "b": "x" * 10, "c": "x" * 10, "d": "x" * 980})
    chunks = split(corpus_files(root), 2)
    assert [len(c.paths) for c in chunks] == [1, 3]  # 1000 | 10+10+980


def test_split_rejects_zero_parts():
    with pytest.raises(ValueError, match="parts"):
        split(["a"], 0)


# --- merge ------------------------------------------------------------------------


def _part(start: int, n: int) -> Partial:
    metas = [DocMeta(start + i, f"d{start + i}", "t", 1) for i in range(n)]
    return Partial(start, {"w": [Posting(start + i, 1) for i in range(n)]}, metas)


def test_merge_concatenates_sorted():
    postings, metas = merge([_part(3, 2), _part(0, 3)])
    assert [p.doc_id for p in postings["w"]] == [0, 1, 2, 3, 4]
    assert [m.doc_id for m in metas] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("parts", [[_part(0, 2), _part(3, 1)], [_part(0, 2), _part(1, 2)]])
def test_merge_rejects_gaps_and_overlaps(parts: list[Partial]):
    with pytest.raises(ValueError, match="contiguous"):
        merge(parts)


def test_merge_empty():
    assert merge([]) == ({}, [])


# --- помилки воркерів: падіння, а не зависання ----------------------------------------


@pytest.mark.parametrize("executor", EXECUTORS)
def test_worker_exception_propagates_with_stack(books: Path, executor: ExecutorKind):
    paths = [*map(str, corpus_files(books)), str(books / "missing.txt")]
    with pytest.raises(WorkerError, match="FileNotFoundError") as info:
        build_parallel(paths, workers=3, executor=executor)
    cause = info.value.__cause__
    assert isinstance(cause, FileNotFoundError)
    if executor == "processes":
        # стек дочірнього процесу: concurrent.futures кладе його текстом у __cause__ винятку
        assert "build_partial" in str(cause.__cause__)
    else:
        # той самий процес: справжній traceback з кадром build_partial
        assert "missing.txt" in str(info.value)


def die(chunk: Chunk, with_positions: bool) -> Partial:
    """Воркер, який «падає по-справжньому» (як segfault або OOM-killer)."""
    os._exit(3)


def test_hard_crash_of_process_raises_not_hangs(books: Path):
    chunks = split(corpus_files(books), 2)
    with make_pool("processes", 2) as pool, pytest.raises(WorkerError, match="died abruptly"):
        map_partials(pool, chunks, task=die)


# --- дрібниці -------------------------------------------------------------------------


def test_partials_are_picklable(books: Path):
    part = build_partial(split(corpus_files(books), 1)[0], True)
    assert pickle.loads(pickle.dumps(part)) == part


def test_on_done_reports_every_chunk(books: Path):
    seen: list[int] = []
    build_parallel(
        corpus_files(books), workers=3, executor="threads", on_done=lambda c: seen.append(c.start)
    )
    assert sorted(seen) == [c.start for c in split(corpus_files(books), 3)]


def test_build_parallel_rejects_zero_workers(books: Path):
    with pytest.raises(ValueError, match="workers"):
        build_parallel(corpus_files(books), workers=0, executor="serial")


def test_runtime_info():
    assert cpu_count() >= 1
    assert isinstance(gil_enabled(), bool)
