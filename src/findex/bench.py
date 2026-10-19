import argparse
import timeit
from pathlib import Path

from findex.boolean import and_merge, and_set, or_merge, or_set
from findex.storage import load


def per_call_us(fn, a, b, number: int) -> float:
    runs = timeit.repeat(lambda: fn(a, b), number=number, repeat=5)
    return min(runs) / number * 1e6  # мікросекунди на один виклик


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("--number", type=int, default=50_000)
    args = ap.parse_args()

    postings, _ = load(args.index)
    frequent = sorted(postings, key=lambda t: (-len(postings[t]), -sum(p.tf for p in postings[t])),
    )[:2]
    rare = sorted(postings, key=lambda t: (len(postings[t]), t))[:2]
    ids = {t: [p.doc_id for p in postings[t]] for t in frequent + rare}

    pairs = [
        ("frequent + frequent", frequent[0], frequent[1]),
        ("frequent + rare", frequent[0], rare[0]),
        ("rare + rare", rare[0], rare[1]),
    ]
    print(f"frequent: {[(t, len(ids[t])) for t in frequent]}")
    print(f"rare:     {[(t, len(ids[t])) for t in rare]}\n")
    print(f"{'pair':<22}{'op':<5}{'merge, us':>11}{'set, us':>10}")
    for name, x, y in pairs:
        a, b = ids[x], ids[y]
        for op, m, s in (("AND", and_merge, and_set), ("OR", or_merge, or_set)):
            tm = per_call_us(m, a, b, args.number)
            ts = per_call_us(s, a, b, args.number)
            print(f"{name:<22}{op:<5}{tm:>11.2f}{ts:>10.2f}")


if __name__ == "__main__":
    main()