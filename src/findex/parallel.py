import gc
import logging
import multiprocessing
import os
import pickle
import sys
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NamedTuple

from findex.corpus import Document, read_text
from findex.index import build_index
from findex.models import DocMeta, Posting
from findex.storage import Postings

log = logging.getLogger(__name__)

type ExecutorKind = Literal["serial", "threads", "processes"]
EXECUTORS: Final[tuple[ExecutorKind, ...]] = ("serial", "threads", "processes")


class Chunk(NamedTuple):
    """Шматок корпусу: файли з doc_id start, start+1, ..., start+len(paths)-1."""

    start: int
    paths: tuple[str, ...]  # str, а не Path: дешевше в pickle і однаково на всіх ОС


class Partial(NamedTuple):
    """Частковий індекс одного шматка."""

    start: int
    postings: Postings
    metas: list[DocMeta]


type Task = Callable[[Chunk, bool], Partial]


class WorkerError(RuntimeError):
    """Воркер упав. Оригінальний виняток (зі стеком воркера) — у __cause__."""


@dataclass(frozen=True, slots=True)
class BuildReport:
    executor: ExecutorKind
    workers: int
    chunks: int
    partial_s: float  # від старту пулу до останнього Partial (включно із запуском процесів)
    unpickle_s: float  # з них: розпаковка результатів у батьківському процесі (лише processes)
    merge_s: float


def cpu_count() -> int:
    """Скільки ядер доступно саме цьому процесу (з урахуванням affinity), мінімум 1."""
    return os.process_cpu_count() or 1


def gil_enabled() -> bool:
    """sys._is_gil_enabled() з'явився в 3.13; у старіших версіях GIL є завжди."""
    probe: Callable[[], bool] | None = getattr(sys, "_is_gil_enabled", None)
    return True if probe is None else probe()


# ------------------------------------------------------------------ split


def split(paths: Sequence[Path | str], parts: int) -> list[Chunk]:
    """Ділить файли на <= parts суцільних шматків, приблизно рівних за байтами.

    Суцільних — бо doc_id це номер файлу в відсортованому списку: шматок
    отримує діапазон [start, start + len) і нумерує свої документи сам, ні з ким
    не домовляючись. Рівних за байтами — бо книжки різного розміру, а збірка
    чекає на найповільніший шматок.
    """
    if parts < 1:
        raise ValueError(f"parts must be >= 1, got {parts}")
    names = [str(p) for p in paths]
    if not names:
        return []
    parts = min(parts, len(names))
    sizes = [max(os.path.getsize(p), 1) if os.path.exists(p) else 1 for p in names]
    total = sum(sizes)

    chunks: list[Chunk] = []
    start = acc = 0
    for i, size in enumerate(sizes):
        acc += size
        left_parts = parts - len(chunks) - 1  # скільки шматків ще треба відкрити після цього
        left_files = len(names) - (i + 1)
        if left_parts == 0:
            break
        if acc >= total * (len(chunks) + 1) / parts or left_files == left_parts:
            chunks.append(Chunk(start, tuple(names[start : i + 1])))
            start = i + 1
    chunks.append(Chunk(start, tuple(names[start:])))
    return chunks


# ------------------------------------------------------------------ worker


def build_partial(chunk: Chunk, with_positions: bool = False) -> Partial:
    """Індекс одного шматка. Функція рівня модуля — її можна віддати процесу.

    Під spawn дочірній процес імпортує findex.parallel заново і знаходить
    build_partial за іменем, тому лямбда чи вкладена функція тут не підійшли б.
    Жодного спільного стану: усе, що функція створює, — локальне.
    """
    docs = (
        Document(chunk.start + i, Path(p), read_text(Path(p))) for i, p in enumerate(chunk.paths)
    )
    postings, metas = build_index(docs, with_positions=with_positions)
    return Partial(chunk.start, postings, metas)


# ------------------------------------------------------------------ merge


def merge(partials: Iterable[Partial]) -> tuple[Postings, list[DocMeta]]:
    """Зливає часткові індекси в один. Виконується в одному потоці — послідовна частина.

    Шматки сортуються за start, тому для кожного терміна списки просто
    дописуються один за одним і лишаються відсортованими за doc_id.
    Увага: списки постінгів першого шматка переходять у результат і дописуються
    (без копіювання) — після merge partials повторно не використовуються.
    """
    postings: dict[str, list[Posting]] = {}
    metas: list[DocMeta] = []
    for part in sorted(partials, key=lambda p: p.start):
        if part.start != len(metas):
            raise ValueError(
                f"doc_id ranges are not contiguous: expected {len(metas)}, "
                f"got chunk starting at {part.start}"
            )
        metas.extend(part.metas)
        for term, plist in part.postings.items():
            dst = postings.get(term)
            if dst is None:
                postings[term] = plist
            else:
                dst.extend(plist)
    return postings, metas


# ------------------------------------------------------------------ executors


def make_pool(kind: ExecutorKind, workers: int) -> AbstractContextManager[Executor | None]:
    """Пул для kind; для serial — None (звичайний цикл у поточному потоці)."""
    match kind:
        case "serial":
            return nullcontext(None)
        case "threads":
            return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="findex")
        case "processes":
            # spawn явно: на Windows і macOS він і так типовий, а на Linux типовий fork
            # приховав би помилки (неімпортовані функції, стан, скопійований з батька).
            ctx = multiprocessing.get_context("spawn")
            return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)


def map_partials(
    pool: Executor | None,
    chunks: Sequence[Chunk],
    with_positions: bool = False,
    *,
    on_done: Callable[[Chunk], None] | None = None,
    task: Task = build_partial,
) -> tuple[list[Partial], float]:
    """Запускає task на кожному шматку -> (часткові індекси, секунди на unpickle у батька).

    При першій помилці скасовує ще не початі задачі й кидає WorkerError.
    """
    if pool is None:
        out: list[Partial] = []
        for chunk in chunks:
            out.append(_call(task, chunk, with_positions))
            if on_done:
                on_done(chunk)
        return out, 0.0

    # процесу віддаємо _pickled (результат — bytes), потоку — _direct (результат — Partial)
    run = _pickled if isinstance(pool, ProcessPoolExecutor) else _direct
    futures: dict[Future[Partial | bytes], Chunk] = {
        pool.submit(run, task, chunk, with_positions): chunk for chunk in chunks
    }

    results: list[Partial] = []
    unpickle_s = 0.0
    pending = set(futures)
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            chunk = futures[fut]
            exc = fut.exception()
            if exc is not None:
                for other in pending:
                    other.cancel()  # ще не початі — не почнуться
                raise _worker_error(chunk, exc) from exc
            match fut.result():
                case bytes() as blob:
                    t0 = time.perf_counter()
                    results.append(_unpickle(blob))
                    unpickle_s += time.perf_counter() - t0
                case Partial() as part:
                    results.append(part)
            if on_done:
                on_done(chunk)
    return results, unpickle_s


@contextmanager
def gc_paused() -> Generator[None]:
    """Тимчасово вимикає циклічний GC (лічильники посилань працюють як завжди)."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _direct(task: Task, chunk: Chunk, with_positions: bool) -> Partial | bytes:
    return task(chunk, with_positions)


def _pickled(task: Task, chunk: Chunk, with_positions: bool) -> Partial | bytes:
    """Виконується у воркері-процесі: Partial -> bytes.

    Навіщо самим, якщо ProcessPoolExecutor і так передає результат через pickle?
    Щоб розпаковувати його в батьківському процесі з вимкненим GC (див. _unpickle):
    bytes через трубу йде як один шматок пам'яті і розпаковується майже миттєво.
    """
    part = task(chunk, with_positions)
    with gc_paused():  # і тут: мемо-словник pickle на мільйон об'єктів теж будить GC
        return pickle.dumps(part, protocol=pickle.HIGHEST_PROTOCOL)


def _unpickle(blob: bytes) -> Partial:
    """pickle.loads із вимкненим циклічним GC.

    Розпаковка створює мільйони Posting і кортежів позицій. Кожні ~2000 нових
    контейнерів GC запускає збирання й обходить їх, хоча циклів тут немає —
    на нашому індексі це втричі довше (див. README лаби 5).
    """
    with gc_paused():
        part = pickle.loads(blob)  # наш же воркер щойно це запакував: довірені дані
    if not isinstance(part, Partial):
        raise TypeError(f"worker returned {type(part).__name__}, expected Partial")
    return part


def _call(task: Task, chunk: Chunk, with_positions: bool) -> Partial:
    try:
        return task(chunk, with_positions)
    except Exception as e:
        raise _worker_error(chunk, e) from e


def _worker_error(chunk: Chunk, exc: BaseException) -> WorkerError:
    last = chunk.start + len(chunk.paths) - 1
    if isinstance(exc, BrokenProcessPool):
        what = "worker process died abruptly"
    else:
        what = f"worker failed with {type(exc).__name__}: {exc}"
    return WorkerError(f"{what} (docs {chunk.start}..{last}, first file {chunk.paths[0]})")


def build_parallel(
    paths: Sequence[Path | str],
    *,
    workers: int,
    executor: ExecutorKind,
    with_positions: bool = False,
    on_done: Callable[[Chunk], None] | None = None,
) -> tuple[Postings, list[DocMeta], BuildReport]:
    """split -> build_partial у пулі -> merge. Для serial шматків стільки ж, скільки workers."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    chunks = split(paths, workers)
    n = max(len(chunks), 1)
    log.info(
        "building with %s, %d workers, %d chunks, GIL %s",
        executor,
        n,
        len(chunks),
        "enabled" if gil_enabled() else "disabled",
    )
    t0 = time.perf_counter()
    with make_pool(executor, n) as pool:
        partials, unpickle_s = map_partials(pool, chunks, with_positions, on_done=on_done)
    t1 = time.perf_counter()
    postings, metas = merge(partials)
    t2 = time.perf_counter()
    report = BuildReport(executor, n, len(chunks), t1 - t0, unpickle_s, t2 - t1)
    log.info(
        "partials %.2f s (unpickle %.2f s), merge %.2f s",
        report.partial_s,
        report.unpickle_s,
        report.merge_s,
    )
    return postings, metas, report
