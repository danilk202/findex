import re
from collections.abc import Iterable, Iterator
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


def iter_hits(text: str, wanted: set[str]) -> Iterator[tuple[int, int, str]]:
    # Входження термів: (початок, кінець, нормалізований терм), за зростанням, ліниво.
    # Лаба 7: токенізувати весь текст на кожен запит задорого (War and Peace — 0,25 с),
    # тож спершу швидкий пошук у C (re + IGNORECASE), а токенізуємо лише рядки з кандидатами.
    # Токен не перетинає \n, тому результат той самий, що й при повному проході.
    if not wanted:
        return
    probe = re.compile(
        "|".join(re.escape(w) for w in sorted(wanted, key=len, reverse=True)), re.IGNORECASE
    )
    line_end = -1
    for cand in probe.finditer(text):
        if cand.start() < line_end:
            continue  # цей рядок уже розібрано
        line_start = text.rfind("\n", 0, cand.start()) + 1
        line_end = text.find("\n", cand.end())
        if line_end == -1:
            line_end = len(text)
        for m in WORD.finditer(text, line_start, line_end):
            if (w := normalize(m.group())) in wanted:
                yield m.start(), m.end(), w


def find_hits(text: str, wanted: set[str]) -> list[tuple[int, int, str]]:
    return list(iter_hits(text, wanted))


def best_window(
    text: str, wanted: set[str], width: int = 80
) -> tuple[list[tuple[int, int, str]], int | None]:
    # (входження, номер найкращого або None). Найкраще — у вікні ±width якого найбільше
    # РІЗНИХ термів (при рівності — найраніше).
    hits: list[tuple[int, int, str]] = []

    def distinct_around(i: int) -> int:
        # рахуємо лише ті входження, що ЦІЛКОМ влазять у вікно — їх і покажемо
        s, e, _ = hits[i]
        seen = {hits[i][2]}
        j = i - 1
        while j >= 0 and hits[j][0] >= s - width:
            seen.add(hits[j][2])
            j -= 1
        j = i + 1
        while j < len(hits) and hits[j][1] <= e + width:
            seen.add(hits[j][2])
            j += 1
        return len(seen)

    # Рання зупинка: вікно з УСІМА термами — кращого не буває, а найраніше з таких
    # і є відповіддю. Вікно входження i повне, щойно прийшло входження далі за e_i + width.
    pending = 0
    for hit in iter_hits(text, wanted):
        hits.append(hit)
        while pending < len(hits) - 1 and hit[0] > hits[pending][1] + width:
            if distinct_around(pending) == len(wanted):
                return hits, pending
            pending += 1
    if not hits:
        return hits, None
    return hits, max(range(len(hits)), key=lambda i: (distinct_around(i), -i))


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
    hits, best = best_window(text, set(terms), width)
    if best is None:
        return _clean(text[: 2 * width]) + " …"

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
