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
    """Усі .txt у корпусі у стабільному порядку — порядок визначає doc_id."""
    return sorted(root.rglob("*.txt"))


def iter_documents(root: Path) -> Iterator[Document]:
    """Ліниво читає документи по одному."""
    for doc_id, path in enumerate(corpus_files(root)):
        log.debug("reading %s", path)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("bad encoding in %s, using errors=replace", path)
            text = path.read_text(encoding="utf-8", errors="replace")
        yield Document(doc_id, path, text)
