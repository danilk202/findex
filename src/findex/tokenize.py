import re
import unicodedata
from collections.abc import Iterator

_WORD = re.compile(r"\w+(?:['’ʼ-]\w+)*")


def tokenize(text: str) -> Iterator[str]:
    text = unicodedata.normalize("NFC", text).casefold()

    for m in _WORD.finditer(text):
        yield m.group()
