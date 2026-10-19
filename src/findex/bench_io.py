import argparse
import tempfile
import time
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.storage import load, save


def best(fn, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()

    print(f"{'positions':<10}{'format':<8}{'size, MiB':>10}{'save, s':>9}{'load, s':>9}")
    with tempfile.TemporaryDirectory() as tmp:
        for positions in (False, True):
            postings, metas = build_index(
                iter_documents(args.root), with_positions=positions
            )
            for fmt in ("pickle", "json"):
                path = Path(tmp) / f"idx_{fmt}_{int(positions)}"
                t_save = best(lambda: save(postings, metas, path, fmt))
                t_load = best(lambda: load(path))
                size = path.stat().st_size / 2**20
                print(
                    f"{str(positions):<10}{fmt:<8}{size:>10.1f}{t_save:>9.2f}{t_load:>9.2f}"
                )


if __name__ == "__main__":
    main()