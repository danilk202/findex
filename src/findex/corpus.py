import json
import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path  # файл; для сторінки з JSONL — «локатор» pages.jsonl#17 (номер рядка)
    text: str
    title: str | None = None  # None — назва файлу без розширення


def is_jsonl(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl" and path.is_file()


def _locator(path: Path) -> tuple[Path, int] | None:
    """pages.jsonl#17 -> (pages.jsonl, 17); звичайний файл -> None."""
    file, sep, line = str(path).rpartition("#")
    if sep and line.isdigit() and file.lower().endswith(".jsonl"):
        return Path(file), int(line)
    return None


def corpus_files(root: Path) -> list[Path]:
    """Усі .txt у корпусі у стабільному порядку — порядок визначає doc_id.

    is_file(): каталог з іменем «x.txt» rglob теж знаходить, а читати його не можна.
    """
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def read_text(path: Path) -> str:
    """UTF-8; зіпсовані байти замінюються на U+FFFD з попередженням у лог.

    Розуміє і локатор pages.jsonl#17 — тоді це текст сторінки з 17-го рядка (для сніпетів).
    """
    log.debug("reading %s", path)
    if (loc := _locator(path)) is not None:
        file, line = loc
        with file.open(encoding="utf-8") as f:
            raw = next(islice(f, line - 1, None), None)
        if raw is None:
            raise FileNotFoundError(2, f"no line {line} in", str(file))
        return _page_text(json.loads(raw))
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("bad encoding in %s, using errors=replace", path)
        return path.read_text(encoding="utf-8", errors="replace")


def iter_documents(root: Path) -> Iterator[Document]:
    """Ліниво читає документи по одному: каталог .txt або JSONL від `findex crawl`."""
    if is_jsonl(root):
        yield from iter_jsonl(root)
        return
    for doc_id, path in enumerate(corpus_files(root)):
        yield Document(doc_id, path, read_text(path))


def iter_jsonl(path: Path) -> Iterator[Document]:
    """По рядку за раз: файл може бути більшим за пам'ять. Порожні рядки пропускаються."""
    doc_id = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                # обірваний останній рядок (краулер убили посеред запису) — не кінець світу
                log.warning("%s:%d: bad JSON (%s), skipped", path, lineno, e.msg)
                continue
            yield Document(doc_id, Path(f"{path}#{lineno}"), _page_text(rec), _title(rec))
            doc_id += 1


def _page_text(rec: dict[str, object]) -> str:
    # заголовок теж шукається: «moby dick» знайде сторінку з таким <title>
    return f"{rec.get('title') or ''}\n{rec.get('text') or ''}"


def _title(rec: dict[str, object]) -> str:
    return str(rec.get("title") or rec.get("url") or "untitled")
