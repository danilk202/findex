from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from findex.corpus import read_text
from findex.tokenize import WORD, normalize

type Marks = tuple[str, str]

ANSI: Final[Marks] = ("\033[1;33m", "\033[0m")  # жирний жовтий у терміналі
MARKDOWN: Final[Marks] = ("**", "**")  # для README


def _clean(s: str) -> str:
    # у книгах Gutenberg переноси рядків посеред речень — зводимо всі пробіли до одного
    return " ".join(s.split())


def _word_end(text: str, i: int) -> int:
    while i < len(text) and text[i].isalnum():
        i += 1
    return i


def _word_start(text: str, i: int) -> int:
    while i > 0 and text[i - 1].isalnum():
        i -= 1
    return i


def make_snippet(
    text: str,
    terms: Iterable[str],
    width: int = 80,
    mark: Marks = MARKDOWN,
) -> str:
    """Вікно ±width символів навколо найкращого входження, терми підсвічені.

    Найкраще входження — те, у вікні якого найбільше РІЗНИХ термів запиту
    (при рівності — найраніше). Чиста функція: текст приходить ззовні, I/O — у doc_snippet.
    """
    wanted = set(terms)
    # шукаємо по оригінальному тексту, щоб мати справжні позиції символів
    hits = [
        (m.start(), m.end(), w)
        for m in WORD.finditer(text)
        if (w := normalize(m.group())) in wanted
    ]
    if not hits:
        return _clean(text[: 2 * width]) + " …"

    # входження відсортовані й не перетинаються, тож і starts, і ends зростають
    starts = [s for s, _, _ in hits]
    ends = [e for _, e, _ in hits]

    def distinct_in_window(i: int) -> int:
        # рахуємо лише ті входження, що ЦІЛКОМ влазять у вікно — їх і покажемо
        s, e, _ = hits[i]
        lo = bisect_left(starts, s - width)
        hi = bisect_right(ends, e + width)
        return len({w for _, _, w in hits[lo:hi]})

    best = max(range(len(hits)), key=lambda i: (distinct_in_window(i), -i))
    s, e, _ = hits[best]
    a, b = max(0, s - width), min(len(text), e + width)
    # не різати слова навпіл: зсуваємо межу, лише якщо вона посеред слова
    if a > 0 and text[a - 1].isalnum() and text[a].isalnum():
        a = _word_end(text, a)
    if b < len(text) and text[b - 1].isalnum() and text[b].isalnum():
        b = _word_start(text, b)

    pieces: list[str] = []
    cur = a
    for hs, he, _ in hits:
        if hs >= a and he <= b:
            pieces += [text[cur:hs], mark[0], text[hs:he], mark[1]]
            cur = he
    pieces.append(text[cur:b])

    prefix = "… " if a > 0 else ""
    suffix = " …" if b < len(text) else ""
    return prefix + _clean("".join(pieces)) + suffix


def doc_snippet(
    path: str | Path, terms: Iterable[str], width: int = 80, mark: Marks = MARKDOWN
) -> str:
    text = read_text(Path(path))  # файл або локатор pages.jsonl#17
    return make_snippet(text, terms, width=width, mark=mark)
