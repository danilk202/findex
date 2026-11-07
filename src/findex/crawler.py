import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Iterable
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urlsplit

import httpx

from findex import __version__
from findex.extract import extract
from findex.fetch import FetchResult, RetryPolicy, fetch
from findex.robots import RobotsCache
from findex.urls import canonicalize, host_key

log = logging.getLogger(__name__)
urllog = logging.getLogger("findex.crawl.urls")  # по рядку на кожен URL -> crawl.log

CONTACT: Final = "+https://github.com/danilk202/findex; danilcreativniy@gmail.com"
USER_AGENT: Final = f"findex/{__version__} ({CONTACT})"
MAX_REDIRECTS: Final = 5


@dataclass(frozen=True, slots=True)
class CrawlConfig:
    max_pages: int = 200  # стеля: скільки HTML-сторінок віддати
    concurrency: int = 5  # воркерів і місць у спільному семафорі
    per_host: int = 2  # одночасних запитів до одного хоста
    host_delay: float = 0.25  # мінімальна пауза між стартами запитів до одного хоста, с
    timeout: float = 10.0  # на одну спробу, asyncio.timeout
    max_depth: int | None = None
    allowed_hosts: frozenset[str] | None = None  # None — хости початкових адрес
    user_agent: str = USER_AGENT
    policy: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    depth: int
    status: int
    title: str
    text: str
    nbytes: int
    elapsed_ms: float
    fetched_at: str

    def to_json(self) -> dict[str, object]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "status": self.status,
            "depth": self.depth,
            "bytes": self.nbytes,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "fetched_at": self.fetched_at,
        }


@dataclass(slots=True)
class CrawlStats:
    """Живі лічильники для екрана rich і підсумку. Змінюються лише в циклі подій."""

    pages: int = 0
    requests: int = 0  # спроб HTTP, разом із повторами
    retries: int = 0
    errors: int = 0
    in_flight: int = 0  # запитів, що зараз чекають відповіді
    queued: int = 0  # глибина черги адрес
    robots_blocked: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def rate(self) -> float:
        return self.pages / self.elapsed if self.elapsed > 0 else 0.0


class HostGate:
    """Ліміти одного хоста: семафор на per_host запитів і пауза між їхніми стартами.

    Порядок важливий: спершу місце в хості, потім у спільному семафорі. Навпаки —
    воркер, що чекає на зайнятий хост, тримав би спільне місце, і запити до вільних
    хостів стояли б у черзі за ним.
    """

    def __init__(self, limit: int, delay: float, shared: asyncio.Semaphore, stats: CrawlStats):
        self._sem = asyncio.Semaphore(limit)
        self._lock = asyncio.Lock()
        self._next = 0.0
        self.delay = delay
        self._shared = shared
        self._stats = stats

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None]:
        async with self._sem:
            async with self._lock:
                loop = asyncio.get_running_loop()
                wait = self._next - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._next = loop.time() + self.delay
            async with self._shared:
                self._stats.in_flight += 1
                try:
                    yield
                finally:
                    self._stats.in_flight -= 1

    def pause(self, seconds: float) -> None:
        """Retry-After від сервера: увесь хост чекає, а не лише один запит."""
        self._next = max(self._next, asyncio.get_running_loop().time() + seconds)


class _Stop(Exception):
    """Сигнал TaskGroup: роботу закінчено, скасувати решту воркерів."""


def make_client(
    config: CrawlConfig, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": config.user_agent},
        follow_redirects=False,  # редіректи обробляє краулер: robots, seen, канонізація
        timeout=httpx.Timeout(config.timeout + 5),  # запасний; головний — asyncio.timeout
        limits=httpx.Limits(max_connections=config.concurrency + 4),
        transport=transport,
    )


class Crawler:
    def __init__(
        self,
        seeds: Iterable[str],
        config: CrawlConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        stats: CrawlStats | None = None,
    ) -> None:
        self.config = config
        self.stats = stats or CrawlStats()
        self.seeds = [u for s in seeds if (u := canonicalize(s))]
        if not self.seeds:
            raise ValueError("no valid http(s) seed URL")
        self.allowed = config.allowed_hosts or frozenset(
            urlsplit(u).hostname or "" for u in self.seeds
        )
        self._transport = transport
        # (глибина, черговість у своєму хості, номер, url, редіректів). Обхід у ширину:
        # спершу вся глибина 1, потім 2... А всередині глибини хости чергуються
        # (1-ша адреса кожного хоста, потім 2-га...), інакше 300 сторінок однієї книги
        # стояли б підряд і всі воркери чекали б на ліміт одного хоста.
        self._queue: asyncio.PriorityQueue[tuple[int, int, int, str, int]] = asyncio.PriorityQueue()
        self._seq = 0
        self._per_host_seq: dict[tuple[int, str], int] = {}
        self._out: asyncio.Queue[Page] = asyncio.Queue(maxsize=config.concurrency * 2)
        self._seen: set[str] = set()
        self._hosts: dict[str, HostGate] = {}
        self._shared = asyncio.Semaphore(config.concurrency)
        self._enough = asyncio.Event()
        self.client: httpx.AsyncClient | None = None

    # ---------------------------------------------------------------- публічне

    async def pages(self) -> AsyncGenerator[Page]:
        """Асинхронний генератор сторінок; закінчується на max_pages або коли адреси скінчились.

        Воркери живуть у TaskGroup в окремій задачі _run, а генератор лише бере
        готові сторінки з черги _out. Так вихід із `async for` (break, виняток)
        не опиняється всередині TaskGroup: у finally задача просто скасовується.
        """
        runner = asyncio.create_task(self._run(), name="crawl")
        getter: asyncio.Future[Page] | None = None
        try:
            while True:
                getter = asyncio.ensure_future(self._out.get())
                done, _ = await asyncio.wait({getter, runner}, return_when="FIRST_COMPLETED")
                if getter in done:
                    yield getter.result()
                    continue
                getter.cancel()
                while not self._out.empty():  # те, що встигли покласти перед кінцем
                    yield self._out.get_nowait()
                runner.result()  # помилка воркера -> тут
                return
        finally:
            if getter is not None:
                getter.cancel()
            if not runner.done():
                runner.cancel()
                with suppress(asyncio.CancelledError):
                    await runner

    # ---------------------------------------------------------------- робота

    async def _run(self) -> None:
        # Створення клієнта — синхронне і не безкоштовне: httpx готує SSL-контекст і читає
        # сотні сертифікатів certifi (на Windows ~0.3 с). asyncio debug спіймав це як
        # блокування циклу, тому — в окремому потоці.
        client = await asyncio.to_thread(make_client, self.config, self._transport)
        async with client:  # закриється в кінці, хоч би що сталося
            self.client = client
            robots = RobotsCache(client, self.config.user_agent, self.config.timeout)
            for url in self.seeds:
                self._enqueue(url, 0, 0)
            try:
                async with asyncio.TaskGroup() as tg:
                    for i in range(self.config.concurrency):
                        tg.create_task(self._worker(robots), name=f"worker-{i}")
                    tg.create_task(self._stop_when(self._queue.join()), name="drained")
                    tg.create_task(self._stop_when(self._enough.wait()), name="max-pages")
            except* _Stop:
                pass  # одна з умов кінця: TaskGroup уже скасувала всіх воркерів
            finally:
                robots.close()
        log.info("crawl finished: %d pages, client closed", self.stats.pages)

    @staticmethod
    async def _stop_when(cond: Awaitable[object]) -> None:
        await cond
        raise _Stop

    async def _worker(self, robots: RobotsCache) -> None:
        while True:
            depth, _, _, url, hops = await self._queue.get()
            self.stats.queued = self._queue.qsize()
            try:
                await self._visit(url, depth, hops, robots)
            finally:
                self._queue.task_done()

    async def _visit(self, url: str, depth: int, hops: int, robots: RobotsCache) -> None:
        if self.stats.pages >= self.config.max_pages:
            return
        if not await robots.allowed(url):
            self.stats.robots_blocked += 1
            urllog.info("ROBOTS   disallowed            %s", url)
            return
        gate = await self._gate(url, robots)
        res = await fetch(
            self._client(),
            url,
            timeout=self.config.timeout,
            policy=self.config.policy,
            gate=gate,
            on_retry=self._on_retry,
        )
        self.stats.requests += res.attempts

        if res.status is not None and 300 <= res.status < 400:
            self._redirect(res, depth, hops)
            return
        if not res.ok:
            self.stats.errors += 1
            self._log(res, depth, "ERROR")
            return
        if not res.is_html:
            self._log(res, depth, f"skip {res.content_type or 'no content-type'}")
            return

        html = res.body.decode(_charset(res.content_type), errors="replace")
        # html.parser — чистий Python: на великій сторінці це десятки мілісекунд CPU,
        # тому в окремому потоці, щоб цикл подій тим часом обслуговував інші сокети
        doc = await asyncio.to_thread(extract, html, url)
        # між перевіркою і += немає await, тож два воркери не перескочать стелю разом
        if self.stats.pages >= self.config.max_pages:
            return
        self.stats.pages += 1
        last = self.stats.pages >= self.config.max_pages
        self._log(res, depth, f"page {self.stats.pages}")
        page = Page(
            url=url,
            depth=depth,
            status=res.status or 0,
            title=doc.title,
            text=doc.text,
            nbytes=res.nbytes,
            elapsed_ms=res.elapsed * 1000,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        await self._out.put(page)  # черга обмежена: повільний споживач гальмує воркерів
        if last:
            self._enough.set()  # -> _stop_when -> TaskGroup скасовує воркерів
            return
        if self.config.max_depth is None or depth < self.config.max_depth:
            for link in doc.links:
                self._enqueue(link, depth + 1, 0)

    def _redirect(self, res: FetchResult, depth: int, hops: int) -> None:
        target = canonicalize(res.location, res.url) if res.location else None
        if target is None:
            self.stats.errors += 1
            self._log(res, depth, "ERROR redirect without valid Location")
        elif hops + 1 > MAX_REDIRECTS:
            self.stats.errors += 1
            self._log(res, depth, f"ERROR more than {MAX_REDIRECTS} redirects, stop at {target}")
        elif target in self._seen:
            self._log(res, depth, f"-> {target} (already seen, not fetched again)")
        else:
            self._log(res, depth, f"-> {target}")
            self._enqueue(target, depth, hops + 1)

    # ---------------------------------------------------------------- допоміжне

    def _enqueue(self, url: str, depth: int, hops: int) -> None:
        if url in self._seen or self._enough.is_set():
            return
        if urlsplit(url).hostname not in self.allowed:
            return
        self._seen.add(url)
        self._seq += 1
        turn = self._per_host_seq.get((depth, host := host_key(url)), 0) + 1
        self._per_host_seq[(depth, host)] = turn
        self._queue.put_nowait((depth, turn, self._seq, url, hops))
        self.stats.queued = self._queue.qsize()

    async def _gate(self, url: str, robots: RobotsCache) -> HostGate:
        key = host_key(url)
        gate = self._hosts.get(key)
        if gate is None:
            delay = max(self.config.host_delay, await robots.crawl_delay(url) or 0.0)
            gate = self._hosts.get(key)  # поки чекали robots, інший воркер міг створити
            if gate is None:
                gate = HostGate(self.config.per_host, delay, self._shared, self.stats)
                self._hosts[key] = gate
        return gate

    def _client(self) -> httpx.AsyncClient:
        assert self.client is not None
        return self.client

    def _on_retry(self, url: str, attempt: int, delay: float, reason: str) -> None:
        self.stats.retries += 1
        urllog.info(
            "RETRY    %-12s try %d/%d in %.2fs %s",
            reason,
            attempt + 1,
            self.config.policy.retries + 1,
            delay,
            url,
        )

    def _log(self, res: FetchResult, depth: int, note: str) -> None:
        status = str(res.status) if res.status is not None else (res.error or "?")[:8]
        urllog.info(
            "%-8s %8dB %6.0fms try=%d d=%d %s  %s",
            status,
            res.nbytes,
            res.elapsed * 1000,
            res.attempts,
            depth,
            res.url,
            note
            + (f" {res.error}" if res.error else "")
            + (f" (total {res.total:.2f}s with retries)" if res.attempts > 1 else ""),
        )


def _charset(content_type: str) -> str:
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            try:
                "".encode(value.strip('"'))
            except LookupError:
                break
            return value.strip('"')
    return "utf-8"


async def crawl(
    seeds: Iterable[str],
    config: CrawlConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
) -> AsyncGenerator[Page]:
    """async for page in crawl(["https://..."], CrawlConfig(max_pages=200)): ..."""
    # aclosing: якщо споживач вийде з циклу, внутрішній генератор закриється одразу
    # (скасує воркерів і закриє клієнт), а не колись потім збирачем сміття
    async with aclosing(
        Crawler(seeds, config or CrawlConfig(), transport=transport, stats=stats).pages()
    ) as gen:
        async for page in gen:
            yield page
