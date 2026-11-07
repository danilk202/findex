import asyncio
import logging
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

import httpx
import pytest

from findex.crawler import CrawlConfig, Crawler, CrawlStats, Page, crawl
from findex.extract import Extracted, extract
from findex.fetch import RetryPolicy

ROBOTS = "User-agent: *\nDisallow: /private/\n"


def html(title: str, *links: str, body: str = "text") -> str:
    a = "".join(f'<a href="{href}">{href}</a>' for href in links)
    return f"<html><head><title>{title}</title></head><body><p>{body}</p>{a}</body></html>"


@dataclass
class FakeSite:
    """Сайт у пам'яті для httpx.MockTransport: шлях -> відповідь, плюс лічильники."""

    pages: dict[str, httpx.Response] = field(default_factory=dict)
    latency: float = 0.01
    calls: Counter[str] = field(default_factory=Counter)
    active: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    peak: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    peak_total: int = 0
    total: int = 0
    starts: defaultdict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def page(self, url: str, title: str, *links: str, body: str = "text") -> None:
        self.pages[url] = httpx.Response(
            200, text=html(title, *links, body=body), headers={"content-type": "text/html"}
        )

    async def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        host = request.url.netloc.decode()
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text=ROBOTS)
        self.calls[url] += 1
        self.starts[host].append(time.perf_counter())
        self.active[host] += 1
        self.total += 1
        self.peak[host] = max(self.peak[host], self.active[host])
        self.peak_total = max(self.peak_total, self.total)
        try:
            await asyncio.sleep(self.latency)
            return self.pages.get(url, httpx.Response(404))
        finally:
            self.active[host] -= 1
            self.total -= 1

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def chain_site(n: int, hosts: tuple[str, ...] = ("a.test",)) -> FakeSite:
    """/0 -> /1 -> ... ; кожна сторінка ще посилається на /0 і на сусідній хост."""
    site = FakeSite()
    for i in range(n):
        for h in hosts:
            links = [f"/{i + 1}", "/0", *(f"http://{o}/{i}" for o in hosts if o != h)]
            site.page(f"http://{h}/{i}", f"{h} page {i}", *links)
    return site


def collect(seeds: list[str], config: CrawlConfig, site: FakeSite) -> tuple[list[Page], CrawlStats]:
    stats = CrawlStats()

    async def go() -> list[Page]:
        return [p async for p in crawl(seeds, config, transport=site.transport(), stats=stats)]

    return asyncio.run(go()), stats


FAST = dict(host_delay=0.0, timeout=2.0)


def test_stops_exactly_at_max_pages():
    site = chain_site(100)
    pages, stats = collect(
        ["http://a.test/0"], CrawlConfig(max_pages=17, concurrency=5, **FAST), site
    )
    assert len(pages) == 17 == stats.pages
    assert len({p.url for p in pages}) == 17


def test_every_url_fetched_once():
    site = chain_site(30, hosts=("a.test", "b.test"))
    hosts = frozenset({"a.test", "b.test"})
    config = CrawlConfig(max_pages=1000, concurrency=8, allowed_hosts=hosts, **FAST)
    pages, _ = collect(["http://a.test/0"], config, site)
    assert len(pages) == 60  # усе, що досяжне на обох хостах
    assert max(site.calls.values()) == 1  # seen: жодної адреси двічі


def test_canonical_duplicates_fetched_once():
    site = FakeSite()
    site.page(
        "http://a.test/",
        "home",
        "/x?b=2&a=1",
        "/x?a=1&b=2#frag",
        "HTTP://A.TEST/x?a=1&b=2",
        "//a.test:80/x?a=1&b=2",
    )
    site.page("http://a.test/x?a=1&b=2", "x")
    pages, _ = collect(["http://a.test/"], CrawlConfig(max_pages=50, **FAST), site)
    assert sorted(p.url for p in pages) == ["http://a.test/", "http://a.test/x?a=1&b=2"]
    assert site.calls["http://a.test/x?a=1&b=2"] == 1


def test_robots_disallowed_never_fetched():
    site = FakeSite()
    site.page("http://a.test/", "home", "/private/secret", "/public")
    site.page("http://a.test/public", "public")
    site.page("http://a.test/private/secret", "secret")
    pages, stats = collect(["http://a.test/"], CrawlConfig(**FAST), site)
    assert {p.url for p in pages} == {"http://a.test/", "http://a.test/public"}
    assert site.calls["http://a.test/private/secret"] == 0
    assert stats.robots_blocked == 1


def test_redirect_loop_terminates():
    site = FakeSite()
    site.page("http://a.test/", "home", "/loop/a")
    site.pages["http://a.test/loop/a"] = httpx.Response(302, headers={"Location": "/loop/b"})
    site.pages["http://a.test/loop/b"] = httpx.Response(302, headers={"Location": "/loop/a"})
    pages, _ = collect(["http://a.test/"], CrawlConfig(**FAST), site)
    assert [p.url for p in pages] == ["http://a.test/"]
    assert site.calls["http://a.test/loop/a"] == site.calls["http://a.test/loop/b"] == 1


def test_redirect_chain_is_capped():
    site = FakeSite()
    site.page("http://a.test/", "home", "/r/0")
    for i in range(20):  # нескінченний ланцюжок різних адрес: seen його не зупинить
        site.pages[f"http://a.test/r/{i}"] = httpx.Response(
            302, headers={"Location": f"/r/{i + 1}"}
        )
    _, stats = collect(["http://a.test/"], CrawlConfig(**FAST), site)
    assert sum(1 for u in site.calls if "/r/" in u) == 6  # 1 + MAX_REDIRECTS
    assert stats.errors == 1


def test_redirect_followed_to_page():
    site = FakeSite()
    site.page("http://a.test/", "home", "/old")
    site.pages["http://a.test/old"] = httpx.Response(301, headers={"Location": "/new#x"})
    site.page("http://a.test/new", "new")
    pages, _ = collect(["http://a.test/"], CrawlConfig(**FAST), site)
    assert {p.url for p in pages} == {"http://a.test/", "http://a.test/new"}


def test_per_host_and_global_limits():
    site = chain_site(40, hosts=("a.test", "b.test", "c.test"))
    site.latency = 0.02
    config = CrawlConfig(max_pages=100, concurrency=4, per_host=2, **FAST)
    collect(["http://a.test/0", "http://b.test/0", "http://c.test/0"], config, site)
    assert max(site.peak.values()) <= 2  # ліміт на хост
    assert site.peak_total <= 4  # спільний семафор
    assert site.peak_total >= 3  # і паралельність справді була


def test_min_delay_between_requests_to_host():
    site = chain_site(10)
    config = CrawlConfig(max_pages=8, concurrency=4, per_host=4, host_delay=0.05, timeout=2)
    collect(["http://a.test/0"], config, site)
    starts = sorted(site.starts["a.test"])
    gaps = [b - a for a, b in pairwise(starts)]
    assert min(gaps) >= 0.035  # допуск: таймер циклу подій може спрацювати трохи раніше


def test_out_of_scope_host_not_fetched():
    site = FakeSite()
    site.page("http://a.test/", "home", "http://elsewhere.test/x", "/y")
    site.page("http://a.test/y", "y")
    pages, _ = collect(["http://a.test/"], CrawlConfig(**FAST), site)
    assert {p.url for p in pages} == {"http://a.test/", "http://a.test/y"}
    assert not any("elsewhere" in u for u in site.calls)


def test_max_depth():
    site = chain_site(10)
    pages, _ = collect(["http://a.test/0"], CrawlConfig(max_depth=2, **FAST), site)
    assert sorted(p.depth for p in pages) == [0, 1, 2]


def test_errors_counted_and_non_html_skipped():
    site = FakeSite()
    site.page("http://a.test/", "home", "/missing", "/file.pdf", "/ok")
    site.pages["http://a.test/file.pdf"] = httpx.Response(
        200, content=b"%PDF", headers={"content-type": "application/pdf"}
    )
    site.page("http://a.test/ok", "ok")
    pages, stats = collect(["http://a.test/"], CrawlConfig(**FAST), site)
    assert {p.url for p in pages} == {"http://a.test/", "http://a.test/ok"}
    assert stats.errors == 1  # 404


def test_retry_through_crawler():
    site = FakeSite()
    site.page("http://a.test/", "home", "/flaky")
    responses = [httpx.Response(503), httpx.Response(200, html="<title>back</title>")]

    async def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/flaky":
            site.calls[str(request.url)] += 1
            return responses.pop(0)
        return await site.handler(request)

    policy = RetryPolicy(retries=3, base=0.01, cap=0.02)

    async def go() -> list[Page]:
        config = CrawlConfig(policy=policy, **FAST)
        return [
            p async for p in crawl(["http://a.test/"], config, transport=httpx.MockTransport(flaky))
        ]

    pages = asyncio.run(go())
    assert {p.title for p in pages} == {"home", "back"}


def test_break_early_cancels_workers_and_closes_client():
    site = chain_site(200, hosts=("a.test", "b.test"))
    crawler = Crawler(["http://a.test/0"], CrawlConfig(max_pages=1000, concurrency=10, **FAST),
                      transport=site.transport())  # fmt: skip

    async def go() -> int:
        got = 0
        gen = crawler.pages()
        async for _ in gen:
            got += 1
            if got == 5:
                break
        await gen.aclose()
        before = sum(site.calls.values())
        await asyncio.sleep(0.1)  # якби воркери жили, лічильник ріс би далі
        assert sum(site.calls.values()) == before
        return got

    assert asyncio.run(go()) == 5
    assert crawler.client is not None and crawler.client.is_closed


def test_html_parsed_outside_event_loop_thread(monkeypatch: pytest.MonkeyPatch):
    """Розбір HTML — чистий CPU: він має йти не в потоці циклу подій."""
    import findex.crawler as crawler_mod

    threads: list[str] = []
    real = crawler_mod.extract

    def spy(html: str, url: str) -> Extracted:
        threads.append(threading.current_thread().name)
        return real(html, url)

    monkeypatch.setattr(crawler_mod, "extract", spy)
    site = chain_site(5)
    collect(["http://a.test/0"], CrawlConfig(**FAST), site)
    assert threads and "MainThread" not in threads


def test_no_blocking_in_debug_mode(caplog: pytest.LogCaptureFixture):
    """asyncio debug: на звичайних сторінках жоден крок не тримає цикл довше за 100 мс.

    Сторінки по ~60 КБ — як довга сторінка документації. На мегабайтних сторінках
    навіть to_thread не рятує повністю: потік розбору тримає той самий GIL (див. README).
    """
    site = FakeSite()
    text = " ".join(f"word{i}" for i in range(8_000))  # ~60 КБ
    for i in range(20):
        site.page(f"http://a.test/{i}", f"p{i}", f"/{i + 1}", body=text)

    async def go() -> list[Page]:
        asyncio.get_running_loop().slow_callback_duration = 0.1  # типовий поріг asyncio
        config = CrawlConfig(concurrency=5, per_host=5, **FAST)
        return [p async for p in crawl(["http://a.test/0"], config, transport=site.transport())]

    with caplog.at_level(logging.WARNING, logger="asyncio"):
        pages = asyncio.run(go(), debug=True)
    assert len(pages) == 20
    assert not [r for r in caplog.records if "Executing" in r.getMessage()]


def test_bad_seed():
    with pytest.raises(ValueError, match="seed"):
        Crawler(["mailto:x@y.z", "ftp://a"], CrawlConfig())


# --- extract -------------------------------------------------------------------------


def test_extract_text_title_links():
    page = (
        "<html><head><title> Hello  World </title><style>p{}</style>"
        "<script>var hidden = 1;</script></head><body><h1>Head</h1><p>One &amp; two</p>"
        '<a href="/a#x">a</a><a href="b?z=1&y=2">b</a><a href="mailto:me@x.y">m</a>'
        '<a href="/a">dup</a><a rel="nofollow" href="/nf">nf</a></body></html>'
    )
    doc = extract(page, "http://h.test/dir/")
    assert doc.title == "Hello World"
    assert "One & two" in doc.text and "hidden" not in doc.text and "p{}" not in doc.text
    assert doc.links == ["http://h.test/a", "http://h.test/dir/b?y=2&z=1"]


def test_extract_base_and_meta_nofollow():
    doc = extract('<base href="http://other.test/x/"><a href="y">y</a>', "http://h.test/")
    assert doc.links == ["http://other.test/x/y"]
    doc = extract(
        '<meta name="robots" content="noindex, nofollow"><a href="/y">y</a>', "http://h.test/"
    )
    assert doc.links == []
