import asyncio
import json
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table

from findex.crawler import CrawlConfig, CrawlStats, crawl
from findex.procstats import peak_rss_mib

# у crawl.log: рядки про кожен URL і robots.txt, а також попередження asyncio (debug)
URL_LOGGERS = ("findex.crawl.urls", "asyncio")


@dataclass(frozen=True, slots=True)
class CrawlSummary:
    pages: int
    wall_s: float
    cpu_s: float
    pages_per_s: float
    requests: int
    retries: int
    errors: int
    robots_blocked: int
    peak_rss_mib: float | None
    concurrency: int

    def to_json(self) -> dict[str, object]:
        return {k: getattr(self, k) for k in self.__slots__}


def render(stats: CrawlStats, config: CrawlConfig) -> Table:
    t = Table(title="findex crawl", show_header=False, min_width=40)
    t.add_column(style="bold")
    t.add_column(justify="right")
    t.add_row("сторінок", f"{stats.pages} / {config.max_pages}")
    t.add_row("сторінок/с", f"{stats.rate:.1f}")
    t.add_row("чекають відповіді", f"{stats.in_flight} / {config.concurrency}")
    t.add_row("помилок", str(stats.errors))
    t.add_row("повторів", str(stats.retries))
    t.add_row("черга адрес", str(stats.queued))
    t.add_row("заборонено robots.txt", str(stats.robots_blocked))
    t.add_row("час", f"{stats.elapsed:.1f} с")
    return t


async def run_crawl(
    seeds: list[str],
    config: CrawlConfig,
    out: TextIO,
    *,
    console: Console | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CrawlStats:
    """Пише кожну сторінку в JSONL одразу, як вона прийшла; у пам'яті — лише поточна."""
    stats = CrawlStats()
    live = (
        Live(get_renderable=lambda: render(stats, config), console=console, refresh_per_second=4)
        if console is not None
        else None
    )
    if live:
        live.start()
    try:
        async with aclosing(crawl(seeds, config, transport=transport, stats=stats)) as pages:
            async for page in pages:
                # запис рядка — мікросекунди, цикл подій від цього не стоїть
                out.write(json.dumps(page.to_json(), ensure_ascii=False) + "\n")
                out.flush()  # обірваний краулер лишає валідний JSONL до останньої сторінки
    finally:
        if live:
            live.stop()
    return stats


def crawl_to_file(
    seeds: list[str],
    config: CrawlConfig,
    output: Path,
    log_file: Path,
    *,
    debug: bool = False,
    console: Console | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CrawlSummary:
    handler = _attach_log(log_file)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        with output.open("w", encoding="utf-8", newline="\n") as f:
            stats = asyncio.run(
                _main(seeds, config, f, console=console, transport=transport, debug=debug),
                debug=debug,
            )
    finally:
        _detach_log(handler)
    wall = time.perf_counter() - t0
    return CrawlSummary(
        pages=stats.pages,
        wall_s=round(wall, 3),
        cpu_s=round(time.process_time() - c0, 3),
        pages_per_s=round(stats.pages / wall, 2) if wall else 0.0,
        requests=stats.requests,
        retries=stats.retries,
        errors=stats.errors,
        robots_blocked=stats.robots_blocked,
        peak_rss_mib=peak_rss_mib(),
        concurrency=config.concurrency,
    )


async def _main(
    seeds: list[str],
    config: CrawlConfig,
    out: TextIO,
    *,
    console: Console | None,
    transport: httpx.AsyncBaseTransport | None,
    debug: bool,
) -> CrawlStats:
    if debug:
        # asyncio сам попередить у лог («Executing <Task ...> took 0.25 seconds»),
        # якщо якийсь крок тримав цикл подій довше за 100 мс
        asyncio.get_running_loop().slow_callback_duration = 0.1
        logging.getLogger("findex.crawl.urls").info("DEBUG    asyncio debug mode on")
    return await run_crawl(seeds, config, out, console=console, transport=transport)


def _attach_log(path: Path) -> logging.Handler:
    """crawl.log: рядок на кожен URL + попередження robots/asyncio. Перезаписується."""
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S")
    )
    for name in URL_LOGGERS:
        logging.getLogger(name).addHandler(handler)
    urls = logging.getLogger("findex.crawl.urls")
    urls.setLevel(logging.INFO)
    urls.propagate = False  # тисячі рядків — лише у файл, не в stderr
    return handler


def _detach_log(handler: logging.Handler) -> None:
    for name in URL_LOGGERS:
        logging.getLogger(name).removeHandler(handler)
    handler.close()
