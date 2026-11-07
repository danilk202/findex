import asyncio
from collections import Counter

import httpx
import pytest

from findex.robots import RobotsCache

AGENT = "findex-test"


def make(responses: dict[str, httpx.Response | Exception]) -> tuple[RobotsCache, Counter[str]]:
    calls: Counter[str] = Counter()

    async def handler(request: httpx.Request) -> httpx.Response:
        calls[str(request.url)] += 1
        await asyncio.sleep(0.01)  # щоб одночасні запити справді перетнулися
        item = responses.get(str(request.url), httpx.Response(404))
        if isinstance(item, Exception):
            raise item
        return item

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return RobotsCache(client, AGENT), calls


def test_rules_per_host_and_cached():
    robots, calls = make(
        {
            "http://a.test/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /x/\n"),
            "http://b.test/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /\n"),
        }
    )

    async def go() -> list[bool]:
        checks = [
            robots.allowed("http://a.test/ok"),
            robots.allowed("http://a.test/x/secret"),
            robots.allowed("http://b.test/anything"),
        ] * 10  # 30 одночасних перевірок
        return await asyncio.gather(*checks)

    results = asyncio.run(go())
    assert results[:3] == [True, False, False]
    assert calls == {"http://a.test/robots.txt": 1, "http://b.test/robots.txt": 1}


def test_ports_are_different_hosts():
    robots, calls = make({})
    asyncio.run(robots.allowed("http://a.test:8100/"))
    asyncio.run(robots.allowed("http://a.test:8101/"))
    assert set(calls) == {"http://a.test:8100/robots.txt", "http://a.test:8101/robots.txt"}


@pytest.mark.parametrize(
    ("response", "allowed"),
    [
        (httpx.Response(404), True),  # robots.txt немає -> можна все
        (httpx.Response(403), True),
        (httpx.Response(500), False),  # сервер хворий -> не чіпаємо
        (httpx.Response(503), False),
        (httpx.ConnectError("down"), False),
    ],
)
def test_missing_or_broken_robots(response: httpx.Response | Exception, allowed: bool):
    robots, _ = make({"http://a.test/robots.txt": response})
    assert asyncio.run(robots.allowed("http://a.test/page")) is allowed


def test_crawl_delay():
    robots, _ = make(
        {"http://a.test/robots.txt": httpx.Response(200, text="User-agent: *\nCrawl-delay: 2\n")}
    )
    assert asyncio.run(robots.crawl_delay("http://a.test/")) == 2.0


def test_agent_specific_rules():
    text = f"User-agent: {AGENT}\nDisallow: /only-us/\n\nUser-agent: *\nDisallow: /\n"
    robots, _ = make({"http://a.test/robots.txt": httpx.Response(200, text=text)})

    async def go() -> tuple[bool, bool]:
        return (
            await robots.allowed("http://a.test/page"),
            await robots.allowed("http://a.test/only-us/x"),
        )

    assert asyncio.run(go()) == (True, False)
