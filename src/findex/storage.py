import json
import pickle
from pathlib import Path

from findex.models import DocMeta, Posting


def save_pickle(
    postings: dict[str, list[Posting]], metas: list[DocMeta], path: Path
) -> None:
    with path.open("wb") as f:
        pickle.dump((postings, metas), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> tuple[dict[str, list[Posting]], list[DocMeta]]:
    
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(
    postings: dict[str, list[Posting]], metas: list[DocMeta], path: Path
) -> None:
    data = {
        "metas": [[m.doc_id, m.path, m.title, m.length] for m in metas],
        "postings": {
            term: [[p.doc_id, p.tf, p.positions] for p in plist]
            for term, plist in postings.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> tuple[dict[str, list[Posting]], list[DocMeta]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    metas = [DocMeta(*m) for m in data["metas"]]
    postings = {
        term: [
            Posting(doc_id, tf, tuple(pos) if pos is not None else None)
            for doc_id, tf, pos in plist
        ]
        for term, plist in data["postings"].items()
    }
    return postings, metas


def save(postings, metas, path: Path, fmt: str = "pickle") -> None:
    {"pickle": save_pickle, "json": save_json}[fmt](postings, metas, path)


def load(path: Path):
    """json-файл починається з '{', усе інше вважаємо pickle."""
    with path.open("rb") as f:
        first = f.read(1)
    return load_json(path) if first == b"{" else load_pickle(path)