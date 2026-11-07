import argparse
import html
import posixpath
import random
import re
import socket
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

from findex.corpus import corpus_files, read_text

ROBOTS = "User-agent: *\nDisallow: /private/\n"


@dataclass(frozen=True, slots=True)
class Book:
    slug: str
    title: str
    host: int
    parts: list[str]  # текст кожної сторінки


@dataclass(slots=True)
class Site:
    """Локальне дзеркало корпусу: книги розкладені по N «хостах» (портах), по ~words слів
    на сторінку. Плюс навмисні збої, щоб краулеру було що обробляти."""

    books: list[Book]
    ports: list[int]
    latency: float = 0.05
    jitter: float = 0.05
    limit: int = 6  # більше одночасних запитів на хост -> 429
    slow: float = 5.0  # /slow/* перший раз відповідає так довго
    hits: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    active: defaultdict[int, int] = field(default_factory=lambda: defaultdict(int))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def base(self, host: int) -> str:
        return f"http://127.0.0.1:{self.ports[host]}"

    def first_hit(self, key: str) -> bool:
        with self.lock:
            self.hits[key] += 1
            return self.hits[key] == 1

    @property
    def num_pages(self) -> int:
        return sum(len(b.parts) for b in self.books)


def load_site(corpus: Path, hosts: int, words: int) -> list[Book]:
    books: list[Book] = []
    for i, path in enumerate(corpus_files(corpus)):
        text = read_text(path)
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), path.stem)
        title = title.removeprefix("\ufeff").removeprefix("The Project Gutenberg eBook of ")[:80]
        paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
        parts: list[str] = []
        cur: list[str] = []
        n = 0
        for p in paras:
            cur.append(p)
            n += p.count(" ") + 1
            if n >= words:
                parts.append("\n".join(cur))
                cur, n = [], 0
        if cur:
            parts.append("\n".join(cur))
        books.append(Book(path.stem, title, i % hosts, parts))
    return books


def _page(title: str, body: str, links: list[tuple[str, str]]) -> bytes:
    nav = "".join(f'<li><a href="{html.escape(h)}">{html.escape(t)}</a></li>' for h, t in links)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>body{{max-width:40em}}</style>"
        f"<script>var x = 'не текст сторінки';</script></head><body><h1>{html.escape(title)}</h1>"
        f"{body}<nav><ul>{nav}</ul></nav></body></html>"
    ).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: httpx тримає з'єднання
    site: ClassVar[Site]
    host: ClassVar[int]

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        site = self.site
        with site.lock:
            site.active[self.host] += 1
            busy = site.active[self.host] > site.limit
        try:
            time.sleep(site.latency + random.uniform(0, site.jitter))  # «мережа»
            if busy:  # сервер захищається від надто жадібного клієнта
                self._send(429, b"too many requests", headers={"Retry-After": "1"})
                return
            self._route()
        finally:
            with site.lock:
                site.active[self.host] -= 1

    def _route(self) -> None:
        site, host = self.site, self.host
        raw = urlsplit(self.path).path
        path = posixpath.normpath(re.sub(r"/{2,}", "/", raw)) + ("/" if raw.endswith("/") else "")
        path = path.replace("//", "/")
        if path == "/robots.txt":
            return self._send(200, ROBOTS.encode(), "text/plain; charset=utf-8")
        if path == "/":
            return self._send(200, self._home())
        if m := re.fullmatch(r"/book/([\w-]+)/", path):
            return self._book(m.group(1))
        if m := re.fullmatch(r"/book/([\w-]+)/(\d+)", path):
            return self._part(m.group(1), int(m.group(2)))
        # --- навмисні збої --------------------------------------------------
        if m := re.fullmatch(r"/flaky/(\d+)", path):  # перший раз 503, потім норм
            if site.first_hit(f"{host}{path}"):
                return self._send(503, b"temporarily unavailable")
            return self._send(200, _page(f"Flaky {m.group(1)}", "<p>recovered</p>", []))
        if m := re.fullmatch(r"/busy/(\d+)", path):  # перший раз 429 з Retry-After: 2
            if site.first_hit(f"{host}{path}"):
                return self._send(429, b"slow down", headers={"Retry-After": "2"})
            return self._send(200, _page(f"Busy {m.group(1)}", "<p>served after wait</p>", []))
        if m := re.fullmatch(r"/slow/(\d+)", path):  # перший раз довше за таймаут
            if site.first_hit(f"{host}{path}"):
                time.sleep(site.slow)
            return self._send(200, _page(f"Slow {m.group(1)}", "<p>finally</p>", []))
        if path == "/loop/a":
            return self._send(302, b"", headers={"Location": "/loop/b"})
        if path == "/loop/b":
            return self._send(302, b"", headers={"Location": "/loop/a"})
        if m := re.fullmatch(r"/moved/(\d+)", path):
            book = site.books[int(m.group(1)) % len(site.books)]
            target = f"{site.base(book.host)}/book/{book.slug}/"
            return self._send(301, b"", headers={"Location": target})
        if path.startswith("/private/"):
            return self._send(200, _page("Private", "<p>robots.txt забороняє</p>", []))
        return self._send(404, b"not found")

    def _home(self) -> bytes:
        site, host = self.site, self.host
        # каталог усіх книг на всіх хостах: з першої ж сторінки фронтир розходиться по хостах
        links = [(f"{site.base(b.host)}/book/{b.slug}/", b.title) for b in site.books]
        links += [(f"{site.base(h)}/", f"host {h}") for h in range(len(site.ports)) if h != host]
        if host == 0:  # збої, щоб у crawl.log було що показати
            links += [
                ("/flaky/1", "flaky"),
                ("/busy/1", "busy"),
                ("/slow/1", "slow"),
                ("/loop/a", "loop"),
                ("/gone", "gone"),
                ("/moved/3", "moved"),
                ("/private/notes", "private"),
                ("mailto:someone@example.com", "mail"),
            ]
        return _page(f"findex mirror, host {host}", "<p>Local mirror of the corpus.</p>", links)

    def _book(self, slug: str) -> None:
        book = next((b for b in self.site.books if b.slug == slug and b.host == self.host), None)
        if book is None:
            return self._send(404, b"no such book")
        links = [(f"/book/{slug}/{i}", f"part {i}") for i in range(1, len(book.parts) + 1)]
        self._send(
            200, _page(book.title, f"<p>{len(book.parts)} parts</p>", [*links, ("/", "home")])
        )

    def _part(self, slug: str, n: int) -> None:
        site = self.site
        book = next((b for b in site.books if b.slug == slug and b.host == self.host), None)
        if book is None or not 1 <= n <= len(book.parts):
            return self._send(404, b"no such part")
        body = "".join(f"<p>{html.escape(p)}</p>" for p in book.parts[n - 1].split("\n"))
        links: list[tuple[str, str]] = [(f"/book/{slug}/", "contents"), ("/", "home")]
        if n > 1:
            links.append((f"/book/{slug}/{n - 1}", "prev"))
        if n < len(book.parts):
            links.append((f"/book/{slug}/{n + 1}#top", "next"))  # фрагмент — та сама сторінка
        rng = random.Random(f"{slug}/{n}")  # «див. також»: детерміновано, з різними формами адрес
        for k in range(3):
            other = rng.choice(site.books)
            part = rng.randint(1, len(other.parts))
            url = f"{site.base(other.host)}/book/{other.slug}/{part}"
            if k == 1:
                url = url.replace("/book/", "//book/") + "?b=2&a=1"  # подвійний слеш, query
            if k == 2:
                url = url.replace("http://127.0.0.1", "HTTP://127.0.0.1") + "?a=1&b=2"
            links.append((url, f"see also: {other.title[:30]}"))
        self._send(200, _page(f"{book.title} — part {n}", body, links))

    def _send(
        self,
        status: int,
        body: bytes,
        ctype: str = "text/html; charset=utf-8",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256
    # На Windows SO_REUSEADDR дозволяє ДВОМ серверам слухати той самий порт, і запити
    # тихо йдуть до старого (з уже «використаними» збоями). Тому там — ексклюзивний порт:
    # другий сервер просто не запуститься.
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request: object, client_address: object) -> None:
        # клієнт не дочекався (таймаут) і закрив з'єднання — для нас це норма, не стек у консоль
        exc = sys.exc_info()[1]
        if not isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            print(f"mirror: {client_address}: {exc!r}", file=sys.stderr)


def serve(site: Site) -> list[Server]:
    servers: list[Server] = []
    for host, port in enumerate(site.ports):
        handler = type(f"Handler{host}", (Handler,), {"site": site, "host": host})
        server = Server(("127.0.0.1", port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    return servers


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m findex.mirror")
    p.add_argument("corpus", type=Path)
    p.add_argument("--port", type=int, default=8100, help="перший порт; хости — port..port+N-1")
    p.add_argument("--hosts", type=int, default=4)
    p.add_argument("--words", type=int, default=1500, help="слів на сторінку")
    p.add_argument("--latency", type=float, default=0.05)
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=6, help="одночасних запитів на хост до 429")
    p.add_argument("--slow", type=float, default=5.0)
    args = p.parse_args()

    books = load_site(args.corpus, args.hosts, args.words)
    ports = [args.port + i for i in range(args.hosts)]
    site = Site(books, ports, args.latency, args.jitter, args.limit, args.slow)
    servers = serve(site)
    print(f"mirror: {site.num_pages} pages, {len(books)} books on {len(ports)} hosts", flush=True)
    for port in ports:
        print(f"  http://127.0.0.1:{port}/", flush=True)
    print("ready", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        for s in servers:
            s.shutdown()


if __name__ == "__main__":
    main()
