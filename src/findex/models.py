from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int,...] | None = None


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int