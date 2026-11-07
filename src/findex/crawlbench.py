import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

from findex.parallel import cpu_count

OUT: Final = Path("bench/lab6")
README: Final = Path("README.md")
PORT: Final = 8100
SEED: Final = f"http://127.0.0.1:{PORT}/"

# (назва, concurrency, per_host, delay): той самий seed і max_pages для всіх
CONFIGS: Final = [
    ("c=1", 1, 4, 0.02),
    ("c=5", 5, 4, 0.02),
    ("c=20", 20, 4, 0.02),
    ("c=20, без ліміту на хост", 20, 20, 0.0),
]


def start_mirror(corpus: Path, errlog: Path) -> subprocess.Popen[str]:
    """Свіжий сервер на кожен прогін: навмисні збої (/flaky, /busy, /slow) знову «перші»."""
    env = {**os.environ, "PYTHONUTF8": "1"}
    with errlog.open("w", encoding="utf-8") as err:  # у файл: труба могла б переповнитись
        proc = subprocess.Popen(
            [sys.executable, "-m", "findex.mirror", str(corpus), "--port", str(PORT)],
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            encoding="utf-8",
            env=env,
        )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip() == "ready":
            return proc
    proc.wait()
    raise SystemExit(
        f"mirror failed to start (порт {PORT} зайнятий? закрийте інший findex.mirror):\n"
        + errlog.read_text("utf-8")
    )


def stop(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def one(
    corpus: Path,
    work: Path,
    name: str,
    c: int,
    per_host: int,
    delay: float,
    pages: int,
    debug: bool = False,
) -> dict[str, Any]:
    mirror = start_mirror(corpus, work / "mirror.err")
    try:
        stats, log = work / "stats.json", work / "crawl.log"
        cmd = [
            sys.executable, "-m", "findex", "crawl", SEED,
            "-o", str(work / "pages.jsonl"), "--log", str(log), "--stats-json", str(stats),
            "-n", str(pages), "-c", str(c), "--per-host", str(per_host),
            "--delay", str(delay), "--timeout", "2", "--no-live",
        ] + (["--debug"] if debug else [])  # fmt: skip
        env = {**os.environ, "PYTHONUTF8": "1", "NO_PROXY": "127.0.0.1,localhost"}
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        if r.returncode != 0:
            raise SystemExit(f"crawl failed:\n{r.stderr}")
        result: dict[str, Any] = json.loads(stats.read_text("utf-8"))
        text = log.read_text("utf-8")
        result["blocked_warnings"] = text.count("Executing <")  # asyncio debug: хтось блокував
        result["name"] = name
        if debug:
            shutil.copy(log, work / "crawl-debug.log")
        return result
    finally:
        stop(mirror)


def run(args: argparse.Namespace) -> None:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[dict[str, Any]]] = {}
    for rep in range(args.reps):
        for name, c, per_host, delay in CONFIGS:
            r = one(args.corpus, out, name, c, per_host, delay, args.pages)
            runs.setdefault(name, []).append(r)
            print(
                f"[{rep + 1}/{args.reps}] {name:28} {r['wall_s']:6.2f} s"
                f"  {r['pages_per_s']:6.1f} p/s  errors {r['errors']:3}"
                f"  retries {r['retries']:3}  rss {r['peak_rss_mib']:.0f} MiB",
                flush=True,
            )

    # окремий прогін з asyncio debug (він сповільнює, тому не в таблиці часу):
    # чи хтось тримав цикл подій довше за 100 мс, і лог зі збоями для README
    dbg = one(args.corpus, out, "c=20 debug", 20, 4, 0.02, args.pages, debug=True)
    print(f"debug run: {dbg['blocked_warnings']} slow-callback warnings", flush=True)

    rows: list[dict[str, Any]] = []
    for name, rs in runs.items():
        med = sorted(rs, key=lambda r: r["wall_s"])[len(rs) // 2]  # медіанний прогін цілком
        rows.append(
            {
                "name": name,
                "wall_s": statistics.median(r["wall_s"] for r in rs),
                "pages_per_s": statistics.median(r["pages_per_s"] for r in rs),
                "cpu_s": statistics.median(r["cpu_s"] for r in rs),
                "errors": med["errors"],
                "retries": med["retries"],
                "rss": statistics.median(r["peak_rss_mib"] or 0 for r in rs),
                "pages": med["pages"],
            }
        )

    print("index outside the loop vs inside a coroutine ...", flush=True)
    lag = subprocess.run(
        [sys.executable, "-m", "findex.looplag", str(out / "pages.jsonl")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    info = {
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "available_cores": cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pages": args.pages,
        "seed": SEED,
    }
    (out / "results.json").write_text(
        json.dumps(
            {"machine": info, "rows": rows, "debug_run": dbg, "raw": runs},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    blocks = {
        "table": table(rows),
        "machine": f"- **Процесор:** {info['cpu']}, {info['logical_cores']} логічних ядер\n"
        f"- **ОС:** {info['os']}, Python {info['python']}\n",
        "looplag": lag.stdout.strip()
        + "\n\nВивід asyncio debug:\n\n```\n"
        + "\n".join(_short(line) for line in lag.stderr.splitlines() if "took" in line)
        + "\n```\n",
        "log": "```\n" + failure_lines(out / "crawl-debug.log") + "```\n",
        "debug": f"Прогін з `--debug` (c=20, {dbg['pages']} сторінок, {dbg['wall_s']:.1f} с): "
        f"попереджень asyncio «Executing ... took» — **{dbg['blocked_warnings']}**.\n",
    }
    update_readme(args.readme, blocks)
    print(f"done: {out / 'results.json'}, {args.readme}")


def table(rows: list[dict[str, Any]]) -> str:
    head = (
        "| конфігурація | сторінок | wall, с | сторінок/с | CPU, с | помилок | повторів "
        "| пік RSS, МіБ |\n|---|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    body = "".join(
        f"| {r['name']} | {r['pages']} | {r['wall_s']:.2f} | **{r['pages_per_s']:.1f}** "
        f"| {r['cpu_s']:.2f} | {r['errors']} | {r['retries']} | {r['rss']:.0f} |\n"
        for r in rows
    )
    return head + body


def failure_lines(log: Path) -> str:
    """Рядки crawl.log про навмисні збої: 503, 429, таймаут, петля редіректів, robots, 404."""
    keep = re.compile(
        r"/(flaky|busy|slow|loop|gone|moved|private)/|/gone\b|ROBOTS   http://127\.0\.0\.1:8100/"
    )
    return "".join(line + "\n" for line in log.read_text("utf-8").splitlines() if keep.search(line))


def _short(line: str) -> str:
    m = re.search(r"coro=<([\w.<>]+)\(\).*took ([\d.]+ seconds)", line)
    return f"Executing <Task ... coro=<{m.group(1)}()> ...> took {m.group(2)}" if m else line


def update_readme(path: Path, blocks: dict[str, str]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- lab6:{name}:start -->\n).*?(<!-- lab6:{name}:end -->)", re.S)
        text = pat.sub(lambda m, b=body: m.group(1) + b + m.group(2), text)
    path.write_text(text, encoding="utf-8")


def _cpu_model() -> str:
    try:
        from findex.parbench import machine_info

        return str(machine_info()["cpu"])
    except Exception:
        return platform.processor() or "unknown"


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m findex.crawlbench")
    p.add_argument("corpus", type=Path, help="каталог .txt для локального дзеркала")
    p.add_argument("--pages", type=int, default=300, help="max_pages (>= 200)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--readme", type=Path, default=README)
    args = p.parse_args()
    t0 = time.perf_counter()
    run(args)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
