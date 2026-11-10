"""Заміри лаби 7: def проти async def і навантаження oha. Запуск — у README."""

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Final

import httpx
from fastapi import FastAPI, Query
from fastapi.routing import APIRoute

from findex.query import NeedsPositionsError, QuerySyntaxError
from findex.web.api import bad_query
from findex.web.app import create_app
from findex.web.deps import Service
from findex.web.schemas import SearchOut, SearchParams

OUT: Final = Path("bench/lab7")
README: Final = Path("README.md")
QUERY: Final = "whale OR ship"


# ---------------------------------------------------------------- async-варіант


async def search_async(params: Annotated[SearchParams, Query()], service: Service) -> SearchOut:
    # Те саме тіло, що й у findex.web.api.search, але async def: FastAPI виконує його
    # прямо в циклі подій, і поки рахуються бали й сніпети, цикл не робить нічого іншого.
    try:
        data = service.search(params.q, params.scorer, params.k, params.page)
    except (QuerySyntaxError, NeedsPositionsError) as e:
        bad_query(e)
    return SearchOut.from_page(data)


def create_async_app() -> FastAPI:
    # Застосунок з лаби, у якому /search — async def. Маршрути перевіряються по черзі,
    # тож маршрут, вставлений першим, перекриває def-версію з findex.web.api. Лише для замірів.
    app = create_app()
    route = APIRoute("/search", search_async, methods=["GET"], response_model=SearchOut)
    app.router.routes.insert(0, route)
    return app


FACTORIES: Final = {
    "def": "findex.web.app:create_app",
    "async": "findex.webbench:create_async_app",
}


# ---------------------------------------------------------------- сервер


def start_server(index: Path, variant: str, workers: int, port: int) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable, "-m", "uvicorn", FACTORIES[variant], "--factory",
        "--host", "127.0.0.1", "--port", str(port), "--workers", str(workers),
        "--log-level", "warning", "--no-access-log",
    ]  # fmt: skip
    env = {**os.environ, "INDEX_PATH": str(index), "LOG_LEVEL": "WARNING"}
    # Windows: окрема група процесів, щоб зупинити uvicorn разом з воркерами (CTRL_BREAK)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen(cmd, env=env, creationflags=flags)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                # з кількома воркерами /health відповів лише один — дамо індексу дочитатись в усіх
                time.sleep(2 if workers > 1 else 0.2)
                return proc
        except httpx.TransportError:
            pass
        if proc.poll() is not None:
            raise SystemExit(f"server exited with code {proc.returncode}")
        time.sleep(0.2)
    stop(proc)
    raise SystemExit("server did not become healthy in 60 s")


def stop(proc: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------- 20 одночасних


async def burst_round(base: str, path: str, n: int) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=n + 5, max_keepalive_connections=n + 5)
    async with httpx.AsyncClient(base_url=base, timeout=120, limits=limits) as c:
        # з'єднання відкриваємо заздалегідь, щоб у замір не потрапив TCP-handshake
        await asyncio.gather(*(c.get("/health") for _ in range(n + 1)))

        async def timed(p: str) -> float:
            t = time.perf_counter()
            r = await c.get(p)
            r.raise_for_status()
            return time.perf_counter() - t

        t0 = time.perf_counter()
        searches = [asyncio.create_task(timed(path)) for _ in range(n)]
        await asyncio.sleep(0.01)  # пошуки вже на сервері — тепер «легкий» запит
        health = await timed("/health")
        lat = await asyncio.gather(*searches)
        wall = time.perf_counter() - t0
    return {"wall_s": wall, "latencies_s": sorted(lat), "health_s": health}


def burst(args: argparse.Namespace) -> None:
    path = f"/search?{httpx.QueryParams({'q': args.query, 'k': 10})}"
    results: dict[str, Any] = {"query": args.query, "n": args.n, "rounds": args.rounds}
    for i, variant in enumerate(("async", "def")):
        proc = start_server(args.index, variant, 1, args.port + i)
        try:
            base = f"http://127.0.0.1:{args.port + i}"
            # один клієнт на всі запити: httpx.get() щоразу створює новий Client, а це
            # SSL-контекст і сотні сертифікатів — на Windows ~0,4 с (див. лабу 6)
            with httpx.Client(base_url=base, timeout=60) as c:
                c.get(path).raise_for_status()  # прогрів кешів текстів і з'єднання
                single: list[float] = []
                for _ in range(5):
                    t = time.perf_counter()
                    c.get(path).raise_for_status()
                    single.append(time.perf_counter() - t)
            rounds = [asyncio.run(burst_round(base, path, args.n)) for _ in range(args.rounds)]
        finally:
            stop(proc)
        lat = sorted(x for r in rounds for x in r["latencies_s"])
        results[variant] = {
            "single_ms": statistics.median(single) * 1000,
            "wall_ms": statistics.median(r["wall_s"] for r in rounds) * 1000,
            "p50_ms": percentile(lat, 50) * 1000,
            "max_ms": lat[-1] * 1000,
            "health_ms": statistics.median(r["health_s"] for r in rounds) * 1000,
            "rounds": rounds,
        }
        brief = {k: round(v, 1) for k, v in results[variant].items() if k != "rounds"}
        print(f"{variant:>5}: {brief}")
    save(args.out / "burst.json", results)
    update_readme(args.readme, {"burst": burst_table(results), "machine": machine_block()})
    print(burst_table(results))


def burst_table(r: dict[str, Any]) -> str:
    rows = [
        f"Запит `{r['query']}`, k=10, {r['n']} одночасних, медіана з {r['rounds']} раундів, "
        "один воркер uvicorn.",
        "",
        "| маршрут | один запит, мс | 20 запитів разом, мс | p50 у пачці, мс "
        "| найдовший, мс | `/health` під час пачки, мс |",
        "|---|---:|---:|---:|---:|---:|",
    ]  # fmt: skip
    for v, name in (("async", "`async def`"), ("def", "`def` (пул потоків)")):
        x = r[v]
        rows.append(
            f"| {name} | {x['single_ms']:.0f} | {x['wall_ms']:.0f} | {x['p50_ms']:.0f} | "
            f"{x['max_ms']:.0f} | {x['health_ms']:.1f} |"
        )
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------- oha


LOCAL: Final = [
    ("1 воркер, `async def`", "async", 1),
    ("1 воркер, `def`", "def", 1),
    ("4 воркери, `def`", "def", 4),
]


def oha(url: str, duration: str, conns: int) -> dict[str, float]:
    exe = shutil.which("oha")
    if exe is None:
        raise SystemExit("oha not found: winget install hatoo.oha (або cargo install oha)")
    # -w: на дедлайні дочекатися запитів у польоті, а не обривати їх. Інакше uvicorn на
    # Windows сиплеться трейсбеками h11 на кожне обірване з'єднання (у статистику вони не йдуть)
    cmd = [exe, "-z", duration, "-c", str(conns), "-w", "--no-tui", "--output-format", "json"]
    cmd.append(url)
    out = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout
    data = json.loads(out)
    pct = data["latencyPercentiles"]
    codes = data.get("statusCodeDistribution", {})
    return {
        "rps": data["summary"]["requestsPerSec"],
        "p50_ms": pct["p50"] * 1000,
        "p95_ms": pct["p95"] * 1000,
        "p99_ms": pct["p99"] * 1000,
        "success": data["summary"]["successRate"],
        "non_200": sum(n for c, n in codes.items() if c != "200"),
    }


def load(args: argparse.Namespace) -> None:
    file = args.out / "load.json"
    rows: dict[str, Any] = json.loads(file.read_text("utf-8")) if file.is_file() else {}
    query = httpx.QueryParams({"q": args.query, "k": 10})
    if not args.skip_local:
        for i, (name, variant, workers) in enumerate(LOCAL):
            proc = start_server(args.index, variant, workers, args.port + i)
            try:
                url = f"http://127.0.0.1:{args.port + i}/search?{query}"
                oha(url, "3s", args.c)  # прогрів: кеш текстів у кожному воркері
                rows[name] = oha(url, args.duration, args.c)
            finally:
                stop(proc)
            print(name, {k: round(v, 1) for k, v in rows[name].items()})
    if args.url:
        url = f"{args.url.rstrip('/')}/search?{query}"
        httpx.get(f"{args.url.rstrip('/')}/health", timeout=120)  # розбудити free-інстанс
        oha(url, "5s", args.c)
        rows["розгорнутий (Render free)"] = oha(url, args.duration, args.c)
        print("deployed", {k: round(v, 1) for k, v in rows["розгорнутий (Render free)"].items()})
    rows["_meta"] = {"query": args.query, "c": args.c, "duration": args.duration}
    save(file, rows)
    update_readme(args.readme, {"load": load_table(rows), "machine": machine_block()})
    print(load_table(rows))


def load_table(rows: dict[str, Any]) -> str:
    meta = rows.get("_meta", {})
    out = [
        f"`oha -z {meta.get('duration', '20s')} -c {meta.get('c', 20)}` на "
        f"`/search?q={meta.get('query', QUERY)}&k=10`.",
        "",
        "| конфігурація | запитів/с | p50, мс | p95, мс | p99, мс |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _, _ in [*LOCAL, ("розгорнутий (Render free)", "", 0)]:
        if (x := rows.get(name)) is None:
            out.append(f"| {name} | — | — | — | — |")
            continue
        bad = f" ({x['non_200']} не 200)" if x.get("non_200") else ""
        out.append(
            f"| {name} | {x['rps']:.1f}{bad} | {x['p50_ms']:.0f} | {x['p95_ms']:.0f} | "
            f"{x['p99_ms']:.0f} |"
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- спільне


def percentile(sorted_values: list[float], p: float) -> float:
    i = min(len(sorted_values) - 1, max(0, round(p / 100 * len(sorted_values)) - 1))
    return sorted_values[i]


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def machine_block() -> str:
    from findex.parbench import machine_info

    m = machine_info()
    ram = f", {m['ram_gib']} ГіБ RAM" if "ram_gib" in m else ""
    return (
        f"- **Процесор:** {m['cpu']}, {m['logical_cores']} логічних ядер{ram}\n"
        f"- **ОС:** {m['os']}, Python {m['python']}\n"
    )


def update_readme(path: Path, blocks: dict[str, str]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- lab7:{name}:start -->\n).*?(<!-- lab7:{name}:end -->)", re.S)
        text = pat.sub(lambda m, b=body: m.group(1) + b + m.group(2), text)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m findex.webbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("burst", "load"):
        s = sub.add_parser(name)
        s.add_argument(
            "index", type=Path, nargs="?", default=Path("index.bin"), help="типово index.bin"
        )
        s.add_argument("--query", default=QUERY)
        s.add_argument("--port", type=int, default=8077)
        s.add_argument("--out", type=Path, default=OUT)
        s.add_argument("--readme", type=Path, default=README)
    b = sub.choices["burst"]
    b.add_argument("-n", type=int, default=20, help="одночасних запитів")
    b.add_argument("--rounds", type=int, default=5)
    ld = sub.choices["load"]
    ld.add_argument("--url", help="адреса розгорнутого сервісу, напр. https://findex.onrender.com")
    ld.add_argument("--skip-local", action="store_true", help="лише --url")
    ld.add_argument("--duration", default="20s")
    ld.add_argument("-c", type=int, default=20, help="одночасних з'єднань oha")
    args = p.parse_args()
    if args.cmd == "burst":
        burst(args)
    else:
        load(args)


if __name__ == "__main__":
    main()
