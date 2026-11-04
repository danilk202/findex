import argparse
import threading
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from findex.parallel import gil_enabled


class SlowHandler(BaseHTTPRequestHandler):
    latency: ClassVar[float] = 0.1
    body: ClassVar[bytes] = b"x" * 64 * 1024

    def do_GET(self) -> None:
        time.sleep(self.latency)
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, format: str, *args: object) -> None:  # тиша в stderr
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 128  # типові 5: при 16+ одночасних з'єднаннях SYN губляться (+1 с)
    daemon_threads = True


# без проксі з оточення: до 127.0.0.1 треба йти напряму
_direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url: str) -> int:
    opener = _direct if url.startswith("http://127.0.0.1") else urllib.request.build_opener()
    with opener.open(url, timeout=30) as r:
        return len(r.read())


def measure(urls: list[str], workers: int, fn: Callable[[str], int] = fetch) -> tuple[float, float]:
    t0, c0 = time.perf_counter(), time.process_time()
    if workers == 1:
        total = sum(fn(u) for u in urls)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            total = sum(pool.map(fn, urls))
    assert total > 0
    return time.perf_counter() - t0, time.process_time() - c0


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m findex.iobench")
    p.add_argument("--n", type=int, default=40, help="скільки запитів")
    p.add_argument("--latency", type=float, default=0.1, help="затримка сервера, с")
    p.add_argument("--urls", type=Path, help="файл зі справжніми URL")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = p.parse_args()

    server = None
    if args.urls:
        urls = [u.strip() for u in args.urls.read_text("utf-8").splitlines() if u.strip()]
        source = f"{len(urls)} URL з {args.urls}"
    else:
        SlowHandler.latency = args.latency
        server = Server(("127.0.0.1", 0), SlowHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        urls = [f"http://127.0.0.1:{port}/doc/{i}" for i in range(args.n)]
        source = f"{args.n} запитів до локального сервера, затримка {args.latency * 1000:.0f} мс"

    print(f"Ввід-вивід: {source}; GIL {'увімкнено' if gil_enabled() else 'вимкнено'}\n")
    print("| потоків | wall, с | CPU, с | прискорення |\n|--:|--:|--:|--:|")
    base = None
    for w in args.workers:
        wall, cpu = measure(urls, w)
        base = base or wall
        print(f"| {w} | {wall:.2f} | {cpu:.2f} | **{base / wall:.1f}×** |", flush=True)
    if server:
        server.shutdown()


if __name__ == "__main__":
    main()
