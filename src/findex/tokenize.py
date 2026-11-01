import re
import unicodedata
from collections.abc import Iterator
from typing import Final

# Слово: \w-символи, всередині можуть бути апостроф або дефіс (don't, п'ять, well-known).
WORD: Final = re.compile(r"\w+(?:['’ʼ-]\w+)*")


def normalize(word: str) -> str:
    """NFC + casefold — одна нормалізація для індексу, запитів і сніпетів."""
    return unicodedata.normalize("NFC", word).casefold()


def tokenize(text: str) -> Iterator[str]:
    """Ліниво видає нормалізовані токени тексту."""
    for m in WORD.finditer(normalize(text)):
        yield m.group()
