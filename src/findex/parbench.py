# pyright: basic
# ^ matplotlib типізований частково (**kwargs: Unknown), а strict вимагає повних типів;
#   решта пакета лишається strict. Цей модуль — лише скрипт замірів.
import argparse
import json
import os
import pickle
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, TypedDict, cast

from findex.corpus import corpus_files
from findex.parallel import (
    EXECUTORS,
    ExecutorKind,
    build_partial,
    cpu_count,
    gc_paused,
    gil_enabled,
    make_pool,
    map_partials,
    merge,
    split,
)

OUT: Final = Path("bench/lab5")
README: Final = Path("README.md")


class Run(TypedDict):
    python: str
    gil: bool
    executor: str
    workers: int
    wall_s: float
    cpu_s: float
    rss_mib: float | None
    unpickle_s: float
    merge_s: float


# ------------------------------------------------------------------ вимірювання в процесі


def _children_cpu() -> float:
    """CPU дочірніх процесів (воркерів), поки вони ще живі. Без psutil — 0."""
    try:
        import psutil
    except ImportError:
        return 0.0
    total = 0.0
    for child in psutil.Process().children(recursive=True):
        try:
            t = child.cpu_times()
            total += t.user + t.system
        except psutil.Error:  # процес уже завершився
            pass
    return total


def _peak_rss_mib() -> float | None:
    """Пік resident set size цього (батьківського) процесу."""
    if sys.platform != "win32":
        import resource  # лише Unix

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # macOS: байти
    try:
        import psutil  # Windows: peak working set = пік RSS

        info = cast(Any, psutil.Process().memory_info())
        return float(info.peak_wset) / 2**20
    except (ImportError, AttributeError):
        pass
    return _win_peak_wset_mib()


def _win_peak_wset_mib() -> float | None:
    """Без psutil (напр. під 3.13t на Windows): GetProcessMemoryInfo через ctypes."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    windll = cast(Any, ctypes).windll
    handle = windll.kernel32.GetCurrentProcess()
    if not windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):
        return None
    return c.PeakWorkingSetSize / 2**20


def one(corpus: Path, executor: ExecutorKind, workers: int, positions: bool) -> Run:
    """Один замір повної збірки: split -> пул -> build_partial -> merge."""
    t0, c0 = time.perf_counter(), time.process_time()
    chunks = split(corpus_files(corpus), workers)
    with make_pool(executor, len(chunks)) as pool:
        partials, unpickle_s = map_partials(pool, chunks, positions)
        child_cpu = _children_cpu()  # до shutdown: після нього воркерів уже немає
    t1 = time.perf_counter()
    merge(partials)
    t2 = time.perf_counter()
    return Run(
        python=platform.python_version() + ("t" if not gil_enabled() else ""),
        gil=gil_enabled(),  # після всіх імпортів: C-модуль без підтримки 3.13t вмикає GIL
        executor=executor,
        workers=len(chunks),
        wall_s=t2 - t0,
        cpu_s=time.process_time() - c0 + child_cpu,
        rss_mib=_peak_rss_mib(),
        unpickle_s=unpickle_s,
        merge_s=t2 - t1,
    )


def pickle_cost(corpus: Path, positions: bool) -> dict[str, float]:
    """Скільки коштує передати весь індекс через pickle (як Partial із воркера)."""
    part = build_partial(split(corpus_files(corpus), 1)[0], positions)
    t0 = time.perf_counter()
    blob = pickle.dumps(part, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.perf_counter()
    pickle.loads(blob)
    t2 = time.perf_counter()
    with gc_paused():
        t3 = time.perf_counter()
        pickle.dumps(part, protocol=pickle.HIGHEST_PROTOCOL)
        t4 = time.perf_counter()
        pickle.loads(blob)
        t5 = time.perf_counter()
    return {
        "mib": len(blob) / 2**20,
        "dumps_s": t1 - t0,
        "loads_s": t2 - t1,
        "dumps_nogc_s": t4 - t3,
        "loads_nogc_s": t5 - t4,
    }


# ------------------------------------------------------------------ драйвер


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "available_cores": cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        return out.stdout.strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _sub(python: str, *args: str, timeout: float = 1800) -> str:
    env = {**os.environ, "PYTHONUTF8": "1"}
    env.pop("PYTHON_GIL", None)  # не дати змінній середовища тихо ввімкнути GIL у 3.13t
    r = subprocess.run(
        [python, "-m", *args],
        capture_output=True,
        encoding="utf-8",  # дочірній пише UTF-8 (PYTHONUTF8=1), а типове на Windows — cp1251
        timeout=timeout,
        env=env,
    )
    if r.returncode != 0:
        raise SystemExit(f"{python} -m {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def _one_sub(python: str, corpus: Path, ex: str, w: int, positions: bool) -> Run:
    pos = "--positions" if positions else "--no-positions"
    out = _sub(python, "findex.parbench", "one", str(corpus), "-x", ex, "-w", str(w), pos)
    return cast(Run, json.loads(out))


def drive(args: argparse.Namespace) -> None:
    corpus: Path = args.corpus
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ncpu = cpu_count()
    workers = sorted({1, 2, 4, 8, ncpu} if args.workers is None else set(args.workers))
    interps: list[tuple[str, str]] = [("gil", sys.executable)]
    if args.ft:
        interps.append(("ft", str(args.ft)))

    # (інтерпретатор, виконавець, воркери). serial — один раз, базова лінія.
    configs: list[tuple[str, str, ExecutorKind, int]] = []
    for tag, py in interps:
        configs.append((tag, py, "serial", 1))
        for kind in ("threads", "processes"):
            if kind == "processes" and tag == "ft" and not args.ft_processes:
                continue  # процеси на 3.13t нічого нового не показують
            for w in workers:
                configs.append((tag, py, kind, int(w)))

    print(f"warm-up (файли в кеш ОС), {len(configs)} конфігурацій x {args.reps}", flush=True)
    _one_sub(sys.executable, corpus, "serial", 1, args.positions)

    runs: dict[tuple[str, str, int], list[Run]] = {}
    for rep in range(args.reps):  # повтори перемежовані: дрейф частоти/температури розмазується
        for tag, py, ex, w in configs:
            r = _one_sub(py, corpus, ex, w, args.positions)
            runs.setdefault((tag, ex, w), []).append(r)
            print(
                f"[{rep + 1}/{args.reps}] {tag:3} {ex:9} w={w:<3} wall {r['wall_s']:6.2f} s "
                f"cpu {r['cpu_s']:6.2f} s  gil={r['gil']}",
                flush=True,
            )

    base = statistics.median(r["wall_s"] for r in runs[("gil", "serial", 1)])
    rows: list[dict[str, Any]] = []
    for (tag, ex, w), rs in runs.items():

        def med(key: str, rs: list[Run] = rs) -> float:
            vals = [float(cast(Any, r)[key]) for r in rs if cast(Any, r)[key] is not None]
            return statistics.median(vals) if vals else float("nan")

        rows.append(
            {
                "interp": tag,
                "python": rs[0]["python"],
                "gil": rs[0]["gil"],
                "executor": ex,
                "workers": w,
                "n": len(rs),
                "wall_s": med("wall_s"),
                "cpu_s": med("cpu_s"),
                "rss_mib": med("rss_mib"),
                "unpickle_s": med("unpickle_s"),
                "merge_s": med("merge_s"),
                "speedup": base / med("wall_s"),
            }
        )

    print("pickle cost ...", flush=True)
    pk = cast(
        dict[str, float],
        json.loads(
            _sub(
                sys.executable,
                "findex.parbench",
                "pickle",
                str(corpus),
                *(["--positions"] if args.positions else []),
            )
        ),
    )
    race = {tag: _sub(py, "findex.race", str(corpus)) for tag, py in interps}
    io = _sub(sys.executable, "findex.iobench") if not args.no_io else ""

    machine = machine_info()
    if args.ft:
        machine["python_ft"] = json.loads(_sub(str(args.ft), "findex.parbench", "version"))
    result = {
        "machine": machine,
        "positions": args.positions,
        "rows": rows,
        "pickle": pk,
        "raw": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in runs.items()},
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot(rows, machine, out_dir / "speedup.png")
    if args.readme.is_file():
        update_readme(args.readme, machine, rows, pk, race, io)
    print(f"done: {out_dir / 'results.json'}, {out_dir / 'speedup.png'}, {args.readme}")


# ------------------------------------------------------------------ вивід


LABELS: Final = {
    ("gil", "serial"): "serial (3.13)",
    ("gil", "threads"): "threads (3.13, GIL)",
    ("gil", "processes"): "processes (3.13, GIL)",
    ("ft", "serial"): "serial (3.13t)",
    ("ft", "threads"): "threads (3.13t, no GIL)",
    ("ft", "processes"): "processes (3.13t)",
}


def table(rows: list[dict[str, Any]]) -> str:
    head = (
        "| Python | GIL | executor | workers | wall, с | CPU, с | CPU / wall | пік RSS, МіБ "
        "| unpickle, с | merge, с | прискорення |\n"
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    order = {"serial": 0, "threads": 1, "processes": 2}
    lines = []
    for r in sorted(rows, key=lambda r: (r["interp"] != "gil", order[r["executor"]], r["workers"])):
        lines.append(
            f"| {r['python']} | {'так' if r['gil'] else '**ні**'} | {r['executor']} "
            f"| {r['workers']} | {r['wall_s']:.2f} | {r['cpu_s']:.2f} "
            f"| {r['cpu_s'] / r['wall_s']:.2f} | {_num(r['rss_mib'], '.0f')} "
            f"| {r['unpickle_s']:.2f} "
            f"| {r['merge_s']:.2f} | **{r['speedup']:.2f}×** |"
        )
    return head + "\n".join(lines) + "\n"


def _num(x: float | None, fmt: str) -> str:
    return "—" if x is None or x != x else format(x, fmt)  # x != x: NaN


def machine_md(m: dict[str, Any]) -> str:
    cores = f"{m.get('physical_cores', '?')} фізичних / {m['logical_cores']} логічних"
    if m["available_cores"] != m["logical_cores"]:
        cores += f" (процесу доступно {m['available_cores']})"
    ram = f", RAM {m['ram_gib']} ГіБ" if "ram_gib" in m else ""
    lines = [
        f"- **Процесор:** {m['cpu']}",
        f"- **Ядра:** {cores}{ram}",
        f"- **ОС:** {m['os']}",
        f"- **Python:** {m['python']} (GIL)"
        + (
            f" і {m['python_ft']['version']} free-threaded, "
            f"`sys._is_gil_enabled()` = `{m['python_ft']['gil']}`"
            if "python_ft" in m
            else ""
        ),
    ]
    return "\n".join(lines) + "\n"


def pickle_md(pk: dict[str, float]) -> str:
    return (
        f"Весь індекс корпусу як один `Partial`: **{pk['mib']:.1f} МіБ** у pickle.\n\n"
        "| | з GC | GC вимкнено |\n|---|--:|--:|\n"
        f"| `pickle.dumps` (у воркері) | {pk['dumps_s']:.2f} с | {pk['dumps_nogc_s']:.2f} с |\n"
        f"| `pickle.loads` (у батька) | {pk['loads_s']:.2f} с | {pk['loads_nogc_s']:.2f} с |\n"
    )


def update_readme(
    path: Path,
    machine: dict[str, Any],
    rows: list[dict[str, Any]],
    pk: dict[str, float],
    race: dict[str, str],
    io: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    blocks = {
        "machine": machine_md(machine),
        "table": table(rows),
        "pickle": pickle_md(pk),
        "race": "\n".join(f"```\n{out.rstrip()}\n```\n" for out in race.values()),
        "io": io,
    }
    for name, body in blocks.items():
        if not body:
            continue
        pat = re.compile(rf"(<!-- lab5:{name}:start -->\n).*?(<!-- lab5:{name}:end -->)", re.S)
        text, n = pat.subn(lambda m, b=body: m.group(1) + b + m.group(2), text)
        if n == 0:
            print(f"README: маркер lab5:{name} не знайдено, пропускаю", file=sys.stderr)
    path.write_text(text, encoding="utf-8")


def plot(rows: list[dict[str, Any]], machine: dict[str, Any], png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {  # категоріальна палітра: фіксований порядок, колір = сутність
        ("gil", "processes"): "#2a78d6",
        ("gil", "threads"): "#eb6834",
        ("ft", "threads"): "#1baf7a",
        ("ft", "processes"): "#6250d6",
    }
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ticks = sorted({r["workers"] for r in rows})
    xmax = ticks[-1]
    ax.plot([1, xmax], [1, xmax], ls="--", lw=1.5, color="#8a8984", label="ідеальне лінійне")
    for key, color in colors.items():
        pts = sorted(
            (r["workers"], r["speedup"]) for r in rows if (r["interp"], r["executor"]) == key
        )
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", ms=6, lw=2, color=color, label=LABELS[key])
    cores = machine.get("physical_cores")
    if cores and cores != machine["logical_cores"]:
        ax.axvline(cores, color="#c3c2b7", lw=1, label=f"{cores} фізичних ядер")
    ft_serial = [r for r in rows if r["interp"] == "ft" and r["executor"] == "serial"]
    if ft_serial:
        ax.axhline(ft_serial[0]["speedup"], color="#1baf7a", lw=1, ls=":", label="serial (3.13t)")
    ax.axhline(1, color="#52514e", lw=1, ls=":", label="serial (3.13) = 1×")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_ylim(0, max(xmax, 2) + 0.5)
    ax.set_xlabel("воркерів")
    ax.set_ylabel("прискорення = wall(serial 3.13) / wall")
    ax.set_title(
        f"findex index: прискорення збірки\n{machine['cpu']}, "
        f"{machine['logical_cores']} логічних ядер",
        fontsize=10,
    )
    ax.grid(True, color="#e4e3df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


# ------------------------------------------------------------------ CLI


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m findex.parbench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("run", help="усі заміри + таблиця, PNG, README (типова команда)")
    d.add_argument("corpus", type=Path)
    d.add_argument("--ft", type=Path, help="python 3.13t (free-threaded) з установленим findex")
    d.add_argument("--ft-processes", action="store_true", help="процеси й на 3.13t")
    d.add_argument("--reps", type=int, default=3)
    d.add_argument("--workers", type=int, nargs="+", help="типово 1 2 4 8 і всі ядра")
    d.add_argument("--positions", action=argparse.BooleanOptionalAction, default=True)
    d.add_argument("--out", type=Path, default=OUT)
    d.add_argument("--readme", type=Path, default=README)
    d.add_argument("--no-io", action="store_true", help="без досліду з вводом-виводом")

    o = sub.add_parser("one", help="(внутрішнє) один замір, JSON у stdout")
    o.add_argument("corpus", type=Path)
    o.add_argument("-x", "--executor", choices=EXECUTORS, required=True)
    o.add_argument("-w", "--workers", type=int, required=True)
    o.add_argument("--positions", action=argparse.BooleanOptionalAction, default=True)

    k = sub.add_parser("pickle", help="(внутрішнє) ціна pickle, JSON у stdout")
    k.add_argument("corpus", type=Path)
    k.add_argument("--positions", action="store_true")

    sub.add_parser("version", help="(внутрішнє) версія й стан GIL")

    # `python -m findex.parbench data/ ...` == `... run data/ ...`
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"run", "one", "pickle", "version", "-h", "--help"}:
        argv.insert(0, "run")
    args = p.parse_args(argv)
    match args.cmd:
        case "one":
            print(json.dumps(one(args.corpus, args.executor, args.workers, args.positions)))
        case "pickle":
            print(json.dumps(pickle_cost(args.corpus, args.positions)))
        case "version":
            print(
                json.dumps(
                    {
                        "version": platform.python_version(),
                        "gil": gil_enabled(),
                        "executable": sys.executable,
                    }
                )
            )
        case "run":
            drive(args)
        case _:
            p.print_help()


if __name__ == "__main__":
    main()
