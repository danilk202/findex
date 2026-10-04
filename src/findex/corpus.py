import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:

    for doc_id, path in enumerate(sorted(root.rglob("*.txt"))):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("bad encoding in %s, using errors=replace", path)
            text = path.read_text(encoding="utf-8", errors="replace")
        yield Document(doc_id, path, text)
