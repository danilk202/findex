import asyncio
import random
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.fetch import FetchResult, RetryPolicy, fetch, parse_retry_after

type Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]
URL = "http://site.test/page"


def scripted(*responses: httpx.Response | Exception | float) -> tuple[Handler, list[str]]:
    """Відповідає по черзі; число — «зависнути» на стільки секунд (для таймаутів)."""
    calls: list[str] = []
    queue = list(responses)

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, float):
            await asyncio.sleep(item)
            return httpx.Response(200, text="late")
        if isinstance(item, Exception):
            raise item
        return item

    return handler, calls


class FakeGate:
    def __init__(self) -> None:
        self.paused: list[float] = []
        self.slots = 0

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None]:
        self.slots += 1
        yield

    def pause(self, seconds: float) -> None:
        self.paused.append(seconds)


def run_fetch(
    handler: Handler, *, timeout: float = 1.0, retries: int = 3, gate: FakeGate | None = None
) -> tuple[FetchResult, list[float]]:
    """fetch через MockTransport; паузи backoff не чекаємо, а записуємо."""
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    async def go() -> FetchResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch(
                client,
                URL,
                timeout=timeout,
                policy=RetryPolicy(retries=retries, base=0.5, cap=10),
                gate=gate,
                rng=random.Random(1),
                sleep=fake_sleep,
            )

    return asyncio.run(go()), slept


def test_ok_records_status_bytes_time():
    handler, calls = scripted(
        httpx.Response(200, html="<p>hello</p>", headers={"content-type": "text/html"})
    )
    res, slept = run_fetch(handler)
    assert (res.status, res.nbytes, res.attempts) == (200, len(b"<p>hello</p>"), 1)
    assert res.ok and res.is_html and res.elapsed >= 0 and res.total >= res.elapsed
    assert calls == [URL] and slept == []


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_other_4xx_not_retried(status: int):
    handler, calls = scripted(httpx.Response(status))
    res, slept = run_fetch(handler)
    assert res.status == status and res.attempts == 1 and len(calls) == 1 and slept == []


def test_5xx_retried_then_ok():
    handler, calls = scripted(httpx.Response(503), httpx.Response(502), httpx.Response(200))
    res, slept = run_fetch(handler)
    assert res.status == 200 and res.attempts == 3 and len(calls) == 3
    assert len(slept) == 2
    assert 0 <= slept[0] <= 0.5 and 0 <= slept[1] <= 1.0  # backoff 0.5·2^n з jitter


def test_gives_up_after_three_retries():
    handler, calls = scripted(httpx.Response(500))
    res, slept = run_fetch(handler)
    assert res.status == 500 and res.attempts == 4 and len(calls) == 4 and len(slept) == 3


def test_retries_zero():
    handler, calls = scripted(httpx.Response(500))
    res, _ = run_fetch(handler, retries=0)
    assert res.attempts == 1 and len(calls) == 1


def test_429_honours_retry_after_and_pauses_host():
    gate = FakeGate()
    handler, _ = scripted(httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200))
    res, slept = run_fetch(handler, gate=gate)
    assert res.status == 200 and res.attempts == 2
    assert slept[0] >= 7  # не раніше, ніж просить сервер
    assert gate.paused == [slept[0]]  # і весь хост чекає разом із нами
    assert gate.slots == 2  # кожна спроба — окремий слот: під час паузи він вільний


def test_429_retry_after_http_date():
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    handler, _ = scripted(httpx.Response(429, headers={"Retry-After": when}), httpx.Response(200))
    _, slept = run_fetch(handler)
    assert 25 <= slept[0] <= 31


def test_retry_after_too_long_gives_up():
    handler, calls = scripted(httpx.Response(503, headers={"Retry-After": "3600"}))
    res, slept = run_fetch(handler)
    assert res.status == 503 and len(calls) == 1 and slept == []


def test_timeout_is_retried_then_ok():
    handler, _ = scripted(5.0, httpx.Response(200, text="ok"))
    res, slept = run_fetch(handler, timeout=0.05)
    assert res.status == 200 and res.attempts == 2 and len(slept) == 1


def test_timeout_every_time():
    handler, _ = scripted(5.0)
    res, _ = run_fetch(handler, timeout=0.02)
    assert res.status is None and res.error == "timeout" and res.attempts == 4


def test_network_error_not_retried():
    handler, calls = scripted(httpx.ConnectError("refused"))
    res, slept = run_fetch(handler)
    assert res.status is None and res.error is not None and res.error.startswith("ConnectError")
    assert len(calls) == 1 and slept == []


def test_redirect_not_followed():
    handler, calls = scripted(httpx.Response(301, headers={"Location": "/new"}))
    res, _ = run_fetch(handler)
    assert res.status == 301 and res.location == "/new" and len(calls) == 1


def test_body_is_capped():
    big = b"x" * 100_000

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big)

    async def go() -> FetchResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch(client, URL, max_bytes=1000)

    res = asyncio.run(go())
    assert res.nbytes == 1000


@pytest.mark.parametrize(
    ("value", "expected"),
    [("3", 3.0), ("0", 0.0), (None, None), ("", None), ("soon", None)],
)
def test_parse_retry_after(value: str | None, expected: float | None):
    assert parse_retry_after(value) == expected


def test_parse_retry_after_past_date_is_zero():
    past = format_datetime(datetime.now(UTC) - timedelta(hours=1), usegmt=True)
    assert parse_retry_after(past) == 0.0


@given(attempt=st.integers(0, 20), seed=st.integers(0, 10_000))
def test_backoff_bounds(attempt: int, seed: int):
    policy = RetryPolicy(base=0.5, cap=10)
    d = policy.backoff(attempt, random.Random(seed))
    assert 0 <= d <= min(10, 0.5 * 2**attempt)
