import argparse
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Final

from findex.corpus import iter_documents
from findex.parallel import gil_enabled
from findex.tokenize import tokenize

REPEATS: Final = 3


def shared_no_lock(slices: list[list[str]]) -> int:
    """Один спільний Counter, без блокування — так робити НЕ можна."""
    counts: Counter[str] = Counter()

    def work(tokens: list[str]) -> None:
        for tok in tokens:
            counts[tok] += 1  # read-modify-write: гонитва

    _run_threads(work, slices)
    return counts.total()


def shared_with_lock(slices: list[list[str]]) -> int:
    """Фікс 1: Lock навколо read-modify-write. Правильно, але потоки стоять у черзі."""
    counts: Counter[str] = Counter()
    lock = threading.Lock()

    def work(tokens: list[str]) -> None:
        for tok in tokens:
            with lock:
                counts[tok] += 1

    _run_threads(work, slices)
    return counts.total()


def local_then_merge(slices: list[list[str]]) -> int:
    """Фікс 2 (як build_partial + merge): у кожного потоку свій Counter, злиття в кінці."""
    partials: list[Counter[str]] = []  # list.append атомарний і в 3.13t (блокування об'єкта)

    def work(tokens: list[str]) -> None:
        partials.append(Counter(tokens))

    _run_threads(work, slices)
    total: Counter[str] = Counter()
    for c in partials:  # merge — в одному потоці
        total.update(c)
    return total.total()


def shared_list_append(slices: list[list[str]]) -> int:
    """Контроль: list.append — одна операція над об'єктом, вона не губиться ніде."""
    seen: list[str] = []

    def work(tokens: list[str]) -> None:
        for tok in tokens:
            seen.append(tok)

    _run_threads(work, slices)
    return len(seen)


def _run_threads(work: Callable[[list[str]], None], slices: list[list[str]]) -> None:
    barrier = threading.Barrier(len(slices))  # стартують одночасно — так гонитва частіша

    def start(tokens: list[str]) -> None:
        barrier.wait()
        work(tokens)

    threads = [threading.Thread(target=start, args=(s,)) for s in slices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m findex.race")
    p.add_argument("corpus", type=Path)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tokens", type=int, default=400_000, help="скільки токенів узяти")
    args = p.parse_args()

    tokens = list(
        islice((t for d in iter_documents(args.corpus) for t in tokenize(d.text)), args.tokens)
    )
    n = args.threads
    slices = [tokens[i::n] for i in range(n)]  # усі потоки б'ють у ті самі часті ключі
    expected = len(tokens)

    print(f"Python {sys.version.split()[0]}, sys._is_gil_enabled() = {gil_enabled()}")
    print(f"{n} потоків, {expected:,} токенів; найгірший з {REPEATS} запусків")
    print(f"{'варіант':<34}{'отримано':>12}{'загублено':>12}{'час, с':>9}")
    variants: list[tuple[str, Callable[[list[list[str]]], int]]] = [
        ("спільний Counter, без Lock", shared_no_lock),
        ("спільний Counter + Lock", shared_with_lock),
        ("свій Counter у потоці + merge", local_then_merge),
        ("спільний list.append (контроль)", shared_list_append),
    ]
    for name, fn in variants:
        worst, slowest = expected, 0.0
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            got = fn(slices)
            slowest = max(slowest, time.perf_counter() - t0)
            worst = min(worst, got)
        print(f"{name:<34}{worst:>12,}{expected - worst:>12,}{slowest:>9.2f}")


if __name__ == "__main__":
    main()
