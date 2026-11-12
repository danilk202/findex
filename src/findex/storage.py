import json
import pickle
from pathlib import Path
from typing import Any, Final, Literal

from findex.arrays import PostingArrays, Postings
from findex.models import DocMeta, Posting

type Format = Literal["pickle", "json"]

# Лаба 8: pickle тепер містить словник масивів NumPy, а не dict[str, list[Posting]].
ARRAYS_TAG: Final = "findex-arrays/1"


class IndexFormatError(ValueError):
    """Файл існує, але це не індекс findex (або він пошкоджений)."""


def save_pickle(postings: Postings, metas: list[DocMeta], path: Path) -> None:
    # формат лаб 2–7: об'єкти Posting як є (лишився для порівняння і старих файлів)
    with path.open("wb") as f:
        pickle.dump((postings, metas), f, protocol=pickle.HIGHEST_PROTOCOL)


def save_arrays(arrays: PostingArrays, metas: list[DocMeta], path: Path) -> None:
    data = {
        "format": ARRAYS_TAG,
        "arrays": arrays.state(),
        # кортежі, а не DocMeta: розпаковка dataclass-а йде через Python-код (див. README)
        "metas": [(m.doc_id, m.path, m.title, m.length) for m in metas],
    }
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _unpickle(path: Path) -> object:
    # УВАГА: pickle.load може виконати довільний код із файлу — лише власні довірені файли.
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
            raise IndexFormatError(f"{path} is not a findex index ({e})") from e


def _arrays_payload(data: object) -> tuple[PostingArrays, list[DocMeta]] | None:
    if not isinstance(data, dict) or data.get("format") != ARRAYS_TAG:  # pyright: ignore[reportUnknownMemberType]
        return None
    payload: dict[str, Any] = data  # pyright: ignore[reportUnknownVariableType]
    try:
        arrays = PostingArrays.from_state(payload["arrays"])
        metas = [DocMeta(int(i), str(p), str(t), int(n)) for i, p, t, n in payload["metas"]]
    except (KeyError, TypeError, ValueError) as e:
        raise IndexFormatError(f"broken findex arrays payload ({e})") from e
    return arrays, metas


def load_pickle(path: Path) -> tuple[Postings, list[DocMeta]]:
    data = _unpickle(path)
    if (found := _arrays_payload(data)) is not None:
        arrays, metas = found
        return arrays.to_postings(), metas
    return _legacy(data, path)


def _legacy(data: object, path: Path) -> tuple[Postings, list[DocMeta]]:
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


def _is_json(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(1) == b"{"


def load(path: Path) -> tuple[Postings, list[DocMeta]]:
    """json-файл починається з '{', усе інше вважаємо pickle."""
    return load_json(path) if _is_json(path) else load_pickle(path)


def load_arrays(path: Path) -> tuple[PostingArrays, list[DocMeta]]:
    # Шлях Index.load: новий формат читається одразу в масиви, старі (pickle лаб 2–7,
    # json) — через об'єкти Posting і одну конвертацію.
    if _is_json(path):
        postings, metas = load_json(path)
    else:
        data = _unpickle(path)
        if (found := _arrays_payload(data)) is not None:
            return found
        postings, metas = _legacy(data, path)
    return PostingArrays.from_postings(postings, [m.length for m in metas]), metas
