import json
import pickle
from pathlib import Path
from typing import Literal

from findex.models import DocMeta, Posting

type Postings = dict[str, list[Posting]]
type Format = Literal["pickle", "json"]


class IndexFormatError(ValueError):
    """Файл існує, але це не індекс findex (або він пошкоджений)."""


def save_pickle(postings: Postings, metas: list[DocMeta], path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump((postings, metas), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> tuple[Postings, list[DocMeta]]:
    # УВАГА: pickle.load може виконати довільний код із файлу — лише власні довірені файли.
    with path.open("rb") as f:
        try:
            data: object = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
            raise IndexFormatError(f"{path} is not a findex index ({e})") from e
    match data:
        # pickle не зберігає типів: перевіряємо форму (dict, list), а типи елементів
        # pyright вивести не може — звідси ignore саме цього правила і лише тут.
        case (dict() as postings, list() as metas):  # pyright: ignore[reportUnknownVariableType]
            return postings, metas  # pyright: ignore[reportUnknownVariableType]
        case _:
            raise IndexFormatError(f"{path} is not a findex index")


def save_json(postings: Postings, metas: list[DocMeta], path: Path) -> None:
    data = {
        "metas": [[m.doc_id, m.path, m.title, m.length] for m in metas],
        "postings": {
            term: [[p.doc_id, p.tf, p.positions] for p in plist] for term, plist in postings.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> tuple[Postings, list[DocMeta]]:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        metas = [DocMeta(int(i), str(p), str(t), int(n)) for i, p, t, n in data["metas"]]
        postings: Postings = {
            str(term): [
                Posting(int(doc_id), int(tf), tuple(pos) if pos is not None else None)
                for doc_id, tf, pos in plist
            ]
            for term, plist in data["postings"].items()
        }
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as e:
        raise IndexFormatError(f"{path} is not a findex index ({e})") from e
    return postings, metas


def save(postings: Postings, metas: list[DocMeta], path: Path, fmt: Format = "pickle") -> None:
    match fmt:
        case "pickle":
            save_pickle(postings, metas, path)
        case "json":
            save_json(postings, metas, path)


def load(path: Path) -> tuple[Postings, list[DocMeta]]:
    """json-файл починається з '{', усе інше вважаємо pickle."""
    with path.open("rb") as f:
        first = f.read(1)
    return load_json(path) if first == b"{" else load_pickle(path)
