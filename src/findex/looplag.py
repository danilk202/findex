import argparse
import asyncio
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index


async def heartbeat(stop: asyncio.Event, lags: list[float], tick: float = 0.01) -> None:
    """Прокидається кожні 10 мс і записує, наскільки запізнився. Запізнення = цикл стояв."""
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        t = loop.time()
        await asyncio.sleep(tick)
        lags.append(loop.time() - t - tick)


def index_terms(path: str) -> int:
    """Для процесу: функція рівня модуля, повертає лише кількість термінів."""
    postings, _ = build_index(iter_documents(Path(path)), with_positions=True)
    return len(postings)


async def scenario(name: str, jsonl: Path) -> tuple[str, float, float]:
    stop = asyncio.Event()
    lags: list[float] = []
    beat = asyncio.create_task(heartbeat(stop, lags))
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    result: object
    if name == "inline":
        # ПОГАНО: CPU-робота прямо в корутині. Поки build_index рахує, жоден інший
        # корутин (воркери краулера, rich, таймаути) не виконується.
        result = build_index(iter_documents(jsonl), with_positions=True)
    elif name == "to_thread":
        # окремий потік — але той самий GIL, тож цикл однаково чекає (див. README)
        result = await asyncio.to_thread(build_index, iter_documents(jsonl), with_positions=True)
    else:
        # окремий процес зі своїм GIL; назад — лише число, а не мільйон об'єктів через pickle
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(1, mp_context=ctx) as pool:
            result = await asyncio.get_running_loop().run_in_executor(pool, index_terms, str(jsonl))
    took = time.perf_counter() - t0
    await asyncio.sleep(0.05)
    stop.set()
    await beat
    del result  # звільнення мільйонів об'єктів індексу — теж CPU; міряємо без нього
    return name, took, max(lags)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m findex.looplag")
    p.add_argument("jsonl", type=Path, help="pages.jsonl від findex crawl")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="asyncio debug: %(message)s")

    async def run() -> list[tuple[str, float, float]]:
        asyncio.get_running_loop().slow_callback_duration = 0.1
        return [await scenario(s, args.jsonl) for s in ("inline", "to_thread", "process")]

    rows = asyncio.run(run(), debug=True)  # debug=True: asyncio сам скаже, хто блокував
    print("| де індексація | час, с | найбільша затримка циклу, мс |")
    print("|---|--:|--:|")
    for name, took, lag in rows:
        where = {
            "inline": "прямо в корутині",
            "to_thread": "asyncio.to_thread",
            "process": "окремий процес (run_in_executor)",
        }[name]
        print(f"| {where} | {took:.2f} | {lag * 1000:.0f} |")


if __name__ == "__main__":  # spawn імпортує цей модуль у дочірньому процесі
    main()
