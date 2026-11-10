"""Корпус для Docker-образу: python -m findex.gutenberg data/ (25 книжок Project Gutenberg)."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Final

import httpx

from findex import __version__

log = logging.getLogger(__name__)

# ім'я файлу в data/ -> номер книжки на gutenberg.org (з рядка «[eBook #N]» у шапці)
BOOKS: Final[dict[str, int]] = {
    "alice": 11,
    "call_wild": 215,
    "christmas_carol": 46,
    "crime_punishment": 2554,
    "dorian_gray": 174,
    "dracula": 345,
    "frankenstein": 84,
    "great_expectations": 1400,
    "huck_finn": 76,
    "jane_eyre": 1260,
    "jekyll_hyde": 43,
    "little_women": 514,
    "moby": 2701,
    "oz": 55,
    "pride": 1342,
    "sherlock": 1661,
    "the_prince": 1232,
    "time_machine": 35,
    "tom_sawyer": 74,
    "treasure_island": 120,
    "two_cities": 98,
    "walden": 205,
    "war_peace": 2600,
    "war_worlds": 36,
    "wuthering": 768,
}
MIRROR: Final = "https://www.gutenberg.org"
USER_AGENT: Final = f"findex/{__version__} (+https://github.com/danilk202/findex)"


def book_url(mirror: str, number: int) -> str:
    return f"{mirror.rstrip('/')}/cache/epub/{number}/pg{number}.txt"


def _get(client: httpx.Client, url: str, retries: int, pause: float) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            r = client.get(url)
            r.raise_for_status()
            return r.content
        except httpx.HTTPError as e:
            if attempt == retries:
                raise
            log.warning("%s: %s, retry %d", url, e, attempt)
            time.sleep(pause * attempt)
    raise ValueError("retries must be >= 1")


def fetch_corpus(
    out: Path,
    *,
    mirror: str = MIRROR,
    client: httpx.Client | None = None,
    retries: int = 3,
    pause: float = 1.0,
) -> list[Path]:
    # Уже наявні файли не перекачуються. Послідовно і з паузою — це чужий сервер.
    out.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60
    )
    written: list[Path] = []
    try:
        for name, number in BOOKS.items():
            path = out / f"{name}.txt"
            if path.is_file() and path.stat().st_size > 0:
                continue
            url = book_url(mirror, number)
            body = _get(client, url, retries, pause)
            path.write_bytes(body)
            written.append(path)
            log.info("%s <- %s (%d KiB)", path, url, len(body) // 1024)
            time.sleep(pause)
    finally:
        if own:
            client.close()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Завантажити 25 книжок корпусу findex")
    parser.add_argument("out", type=Path, help="Каталог для .txt")
    parser.add_argument("--mirror", default=MIRROR, help=f"Типово {MIRROR}")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        written = fetch_corpus(args.out, mirror=args.mirror)
    except httpx.HTTPError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{len(written)} downloaded, {len(BOOKS) - len(written)} already present -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
