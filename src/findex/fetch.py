import asyncio
import email.utils
import logging
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER: Final = 60.0  # якщо сервер просить чекати довше — здаємось, а не висимо


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int | None  # None — відповіді не було (таймаут, мережа)
    nbytes: int
    elapsed: float  # секунди на останню спробу: від запиту до кінця тіла (без черги до хоста)
    attempts: int
    total: float = 0.0  # секунди від виклику fetch: черга до хоста, усі спроби і паузи
    content_type: str = ""
    body: bytes = b""
    location: str | None = None  # для 3xx: куди веде редірект
    error: str | None = None  # "timeout", "ConnectError: ..." тощо

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    retries: int = 3  # повторів, тобто спроб щонайбільше retries + 1
    base: float = 0.5  # перша пауза до jitter
    cap: float = 10.0  # стеля паузи

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """Exponential backoff із «повним» jitter: випадково з [0, min(cap, base·2^n)].

        Jitter потрібен, щоб воркери, які впали одночасно, не повторили теж одночасно
        і не вдарили сервер новою хвилею.
        """
        return rng.uniform(0, min(self.cap, self.base * 2**attempt))


class Gate(Protocol):
    """Що fetch знає про ліміти краулера: слот на один запит і паузу хосту."""

    def slot(self) -> AbstractAsyncContextManager[None]: ...
    def pause(self, seconds: float) -> None: ...


class _NoGate:
    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None]:
        yield

    def pause(self, seconds: float) -> None:
        pass


type OnRetry = Callable[[str, int, float, str], None]  # url, номер повтору, пауза, причина


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Retry-After: або секунди ("3"), або HTTP-дата. Повертає секунди >= 0."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    policy: RetryPolicy | None = None,
    max_bytes: int = 5 * 2**20,
    gate: Gate | None = None,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: OnRetry | None = None,
) -> FetchResult:
    """GET без переходу за редіректами; повтори лише для 429, 5xx і таймауту.

    Кожна спроба — під asyncio.timeout(timeout) і в окремому slot() шлюзу: під час
    паузи між спробами слот звільняється, і інші запити до хоста не стоять.
    Інші 4xx, 3xx і мережеві помилки (крім таймауту) не повторюються.
    """
    gate = gate or _NoGate()
    policy = policy or RetryPolicy()
    rng = rng or random.Random()
    start = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        reason: str
        wait: float | None = None
        t0 = start
        try:
            async with gate.slot():
                t0 = time.perf_counter()  # час рахуємо від моменту, коли слот отримано
                async with asyncio.timeout(timeout):
                    resp, body = await _get(client, url, max_bytes)
        except (TimeoutError, httpx.TimeoutException):
            result = _result(url, t0, start, attempt, error="timeout")
            reason = "timeout"
        except httpx.HTTPError as e:
            return _result(url, t0, start, attempt, error=f"{type(e).__name__}: {e}")
        else:
            result = _result(
                url,
                t0,
                start,
                attempt,
                status=resp.status_code,
                body=body,
                content_type=resp.headers.get("content-type", ""),
                location=resp.headers.get("location"),
            )
            if resp.status_code not in RETRY_STATUSES:
                return result
            reason = str(resp.status_code)
            wait = parse_retry_after(resp.headers.get("retry-after"))

        if attempt > policy.retries:
            return result
        delay = policy.backoff(attempt - 1, rng)
        if wait is not None:
            if wait > MAX_RETRY_AFTER:
                return result  # «приходь через годину» — не наш випадок
            delay = max(delay, wait)  # Retry-After — нижня межа, jitter лише зверху
            gate.pause(delay)  # чекає весь хост, а не лише цей запит
        if on_retry:
            on_retry(url, attempt, delay, reason + (" Retry-After" if wait is not None else ""))
        await sleep(delay)


async def _get(client: httpx.AsyncClient, url: str, max_bytes: int) -> tuple[httpx.Response, bytes]:
    async with client.stream("GET", url) as resp:
        chunks: list[bytes] = []
        size = 0
        async for chunk in resp.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= max_bytes:  # не тягнемо гігабайтний файл заради індексу
                break
        return resp, b"".join(chunks)[:max_bytes]


def _result(
    url: str,
    t0: float,
    start: float,
    attempts: int,
    *,
    status: int | None = None,
    body: bytes = b"",
    content_type: str = "",
    location: str | None = None,
    error: str | None = None,
) -> FetchResult:
    return FetchResult(
        url=url,
        status=status,
        nbytes=len(body),
        elapsed=(now := time.perf_counter()) - t0,
        attempts=attempts,
        total=now - start,
        content_type=content_type,
        body=body,
        location=location,
        error=error,
    )
