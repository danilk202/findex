import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def corpus_files(root: Path) -> list[Path]:
    """Усі .txt у корпусі у стабільному порядку — порядок визначає doc_id.

    is_file(): каталог з іменем «x.txt» rglob теж знаходить, а читати його не можна.
    """
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def read_text(path: Path) -> str:
    """UTF-8; зіпсовані байти замінюються на U+FFFD з попередженням у лог."""
    log.debug("reading %s", path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("bad encoding in %s, using errors=replace", path)
        return path.read_text(encoding="utf-8", errors="replace")


def iter_documents(root: Path) -> Iterator[Document]:
    """Ліниво читає документи по одному."""
    for doc_id, path in enumerate(corpus_files(root)):
        yield Document(doc_id, path, read_text(path))
