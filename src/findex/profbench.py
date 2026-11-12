# Заміри лаби 8: до/після на командах, скорер лаби 3 проти NumPy, сніпети. Запуск — у README.

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import timeit
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Final

from findex.corpus import Document, corpus_files, read_text
from findex.engine import Searcher
from findex.index import Index, build_index
from findex.models import Posting
from findex.query import parse
from findex.ranking import search, search_np
from findex.scoring import BM25
from findex.snippets import find_hits
from findex.tokenize import WORD, normalize

OUT: Final = Path("bench/lab8")
README: Final = Path("README.md")
QUERY3: Final = "whale ship captain"
FREQUENT: Final = "the"
RARE: Final = "harpooneer"
WARM_QUERIES: Final = (QUERY3, "whale OR ship OR captain", "love marriage money", "ghost OR spirit")


# ---------------------------------------------------------------- корпуси


def passages(data: Path, words: int = 500) -> Iterator[Document]:
    # Ті самі книжки, порізані на шматки по `words` слів: ~6 тис. документів
    # і списки постінгів у тисячі елементів замість не більше 25.
    doc_id = 0
    for path in corpus_files(data):
        tokens = read_text(path).split()
        for n, i in enumerate(range(0, len(tokens), words)):
            text = " ".join(tokens[i : i + words])
            yield Document(doc_id, Path(f"{path}#{n}"), text, f"{path.stem} §{n}")
            doc_id += 1


class Lab3Index(Index):
    # Списки Posting зібрані заздалегідь, як dict у лабах 2–7: скорер з лаби 3 міряється
    # на своїй рідній структурі, а не на збиранні Posting з масивів.
    def __init__(self, index: Index) -> None:
        super().__init__(index.arrays, index.metas)
        self._lists = index.arrays.to_postings()

    def __getitem__(self, term: str) -> list[Posting]:
        return self._lists[term]


def per_call_ms(fn: Callable[[], object], repeat: int = 7) -> float:
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat, number)) / number * 1000


# ---------------------------------------------------------------- скорер


def scorer_rows(name: str, index: Index) -> list[dict[str, Any]]:
    lab3 = Lab3Index(index)
    searcher = Searcher(index, BM25())
    rows: list[dict[str, Any]] = []
    for label, query in (("частий", FREQUENT), ("рідкісний", RARE), ("три терми", QUERY3)):
        tree = parse(query)
        terms = list(tree.terms())
        cands = searcher.match(tree)
        a = search(lab3, terms, BM25(), 10, cands)
        b = search_np(index, terms, BM25(), 10, cands)
        assert [(r.doc_id, r.score) for r in a] == [(r.doc_id, r.score) for r in b], query
        py = per_call_ms(lambda t=terms, c=cands: search(lab3, t, BM25(), 10, c))
        vec = per_call_ms(lambda t=terms, c=cands: search_np(index, t, BM25(), 10, c))
        rows.append(
            {
                "corpus": name,
                "docs": index.num_docs,
                "kind": label,
                "query": query,
                "postings": sum(index.df(t) for t in terms),
                "candidates": len(cands),
                "lab3_ms": py,
                "numpy_ms": vec,
                "speedup": py / vec,
            }
        )
        print(rows[-1])
    return rows


def scorer(args: argparse.Namespace) -> None:
    books = Index(*build_index(_books(args.data), with_positions=False))
    parts = Index(*build_index(passages(args.data, args.words), with_positions=False))
    rows = scorer_rows("25 книжок", books) + scorer_rows(f"шматки по {args.words} слів", parts)
    _merge_results(args.out, {"scorer": rows})
    update_readme(args.readme, {"scorer": scorer_table(rows)})


def _books(data: Path) -> Iterator[Document]:
    for i, p in enumerate(corpus_files(data)):
        yield Document(i, p, read_text(p))


def scorer_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| корпус | запит | постінгів | кандидатів | лаба 3, мс | NumPy, мс | прискорення |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['corpus']} ({r['docs']:,} док.) | {r['kind']}: `{r['query']}` "
            f"| {r['postings']:,} | {r['candidates']:,} | {r['lab3_ms']:.3f} "
            f"| {r['numpy_ms']:.3f} | {r['speedup']:.1f}× |".replace(",", " ")
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- сніпети


def iter_hits_lab7(text: str, wanted: set[str]) -> Iterator[tuple[int, int, str]]:
    # Копія iter_hits з лаби 7 — лише для порівняння.
    probe = re.compile(
        "|".join(re.escape(w) for w in sorted(wanted, key=len, reverse=True)), re.IGNORECASE
    )
    line_end = -1
    for cand in probe.finditer(text):
        if cand.start() < line_end:
            continue
        line_start = text.rfind("\n", 0, cand.start()) + 1
        line_end = text.find("\n", cand.end())
        if line_end == -1:
            line_end = len(text)
        for m in WORD.finditer(text, line_start, line_end):
            if (w := normalize(m.group())) in wanted:
                yield m.start(), m.end(), w


def snippets(args: argparse.Namespace) -> None:
    texts = [read_text(p) for p in corpus_files(args.data)]
    queries = [set(parse(q).terms()) for q in WARM_QUERIES]
    for t in texts:  # обидва варіанти — на прогрітому: lower() кешується так само, як на сервері
        find_hits(t, {"whale"})
    rows: list[dict[str, Any]] = []
    for q, wanted in zip(WARM_QUERIES, queries, strict=True):
        old = [list(iter_hits_lab7(t, wanted)) for t in texts]
        new = [find_hits(t, wanted) for t in texts]
        assert old == new, q
        t_old = per_call_ms(lambda w=wanted: [list(iter_hits_lab7(t, w)) for t in texts], 3)
        t_new = per_call_ms(lambda w=wanted: [find_hits(t, w) for t in texts], 3)
        rows.append(
            {
                "query": q,
                "hits": sum(map(len, new)),
                "lab7_ms": t_old,
                "lab8_ms": t_new,
                "speedup": t_old / t_new,
            }
        )
        print(rows[-1])
    _merge_results(args.out, {"snippets": rows})
    lines = [
        "| запит | входжень у 25 книжках | лаба 7, мс | лаба 8, мс | прискорення |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['query']}` | {r['hits']:,} | {r['lab7_ms']:.1f} | {r['lab8_ms']:.1f} "
            f"| {r['speedup']:.1f}× |".replace(",", " ")
        )
    update_readme(args.readme, {"snippets": "\n".join(lines) + "\n"})


# ---------------------------------------------------------------- команди до/після

# Виконується інтерпретатором кожного дерева (до і після), тож лише API, яке є в обох.
PROBE: Final = """
import json, statistics, sys, time, psutil
from findex.index import Index
from findex.web.service import SearchService
path, queries = sys.argv[1], json.loads(sys.argv[2])
loads = []
for _ in range(5):
    t = time.perf_counter(); idx = Index.load(path); loads.append(time.perf_counter() - t)
rss = psutil.Process().memory_info().rss
svc = SearchService(idx)
for q in queries:
    svc.search(q)
times = []
for _ in range(20):
    for q in queries:
        t = time.perf_counter(); svc.search(q); times.append(time.perf_counter() - t)
times.sort()
print(json.dumps({"load_ms": statistics.median(loads) * 1000, "rss_mib": rss / 2**20,
                  "warm_p50_ms": times[len(times) // 2] * 1000,
                  "warm_p95_ms": times[int(len(times) * 0.95) - 1] * 1000}))
"""


def _tree_python(tree: Path) -> list[str]:
    exe = tree / (".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python")
    return [str(exe)]


def _wall(cmd: list[str], cwd: Path, runs: int) -> float:
    times: list[float] = []
    for _ in range(runs):
        t = time.perf_counter()
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def measure_tree(tree: Path, data: Path, runs: int) -> dict[str, Any]:
    py = _tree_python(tree)
    index = (OUT / f"probe-{tree.name}.bin").resolve()
    cli = [*py, "-m", "findex"]
    res: dict[str, Any] = {
        "index_s": _wall([*cli, "index", str(data), "-o", str(index)], tree, max(1, runs // 2)),
        "search_s": _wall([*cli, "search", str(index), QUERY3], tree, runs),
        "file_mib": index.stat().st_size / 2**20,
    }
    probe = subprocess.run(
        [*py, "-c", PROBE, str(index), json.dumps(WARM_QUERIES)],
        cwd=tree,
        check=True,
        capture_output=True,
        text=True,
    )
    res |= json.loads(probe.stdout.strip().splitlines()[-1])
    index.unlink()
    print(tree, res)
    return res


def commands(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    rows = {
        "лаба 7": measure_tree(args.before.resolve(), data, args.runs),
        "лаба 8": measure_tree(Path.cwd(), data, args.runs),
    }
    _merge_results(args.out, {"commands": rows, "machine": machine_block()})
    a, b = rows["лаба 7"], rows["лаба 8"]
    spec = [
        ("`findex index data/` (процеси), с", "index_s", 2),
        (f"`findex search index.bin '{QUERY3}'`, с", "search_s", 2),
        ("`Index.load`, мс", "load_ms", 0),
        ("пам'ять процесу з індексом, МіБ", "rss_mib", 0),
        ("файл індексу, МіБ", "file_mib", 1),
        ("пошук на сервері (прогрітий), p50, мс", "warm_p50_ms", 1),
        ("пошук на сервері (прогрітий), p95, мс", "warm_p95_ms", 1),
    ]
    lines = ["| що міряємо | лаба 7 | лаба 8 | разів |", "|---|---:|---:|---:|"]
    for label, key, nd in spec:
        lines.append(f"| {label} | {a[key]:.{nd}f} | {b[key]:.{nd}f} | {a[key] / b[key]:.1f}× |")
    update_readme(args.readme, {"commands": "\n".join(lines) + "\n", "machine": machine_block()})


# ---------------------------------------------------------------- спільне


def machine_block() -> str:
    from findex.webbench import machine_block as mb

    return mb()


def _merge_results(out: Path, part: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data |= part
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def update_readme(path: Path, blocks: dict[str, str]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- lab8:{name}:start -->\n).*?(<!-- lab8:{name}:end -->)", re.S)
        text = pat.sub(lambda m, b=body: m.group(1) + b + m.group(2), text)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m findex.profbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("scorer", "snippets", "commands"):
        s = sub.add_parser(name)
        s.add_argument("data", type=Path, nargs="?", default=Path("data"))
        s.add_argument("--out", type=Path, default=OUT)
        s.add_argument("--readme", type=Path, default=README)
    sub.choices["scorer"].add_argument("--words", type=int, default=500)
    c = sub.choices["commands"]
    c.add_argument("--before", type=Path, required=True, help="git worktree з тегом lab-07")
    c.add_argument("--runs", type=int, default=5)
    args = p.parse_args()
    {"scorer": scorer, "snippets": snippets, "commands": commands}[args.cmd](args)


if __name__ == "__main__":
    main()
