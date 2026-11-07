from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Final

from findex.urls import canonicalize

_SKIP: Final = frozenset({"script", "style", "noscript", "template", "svg", "head"})
_BLOCK: Final = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
     "section", "article", "header", "footer", "pre", "blockquote", "title", "nav", "table"}
)  # fmt: skip
# між сусідніми посиланнями/кнопками в розмітці часто немає пробілу: <a>x</a><a>y</a>
_GAP: Final = frozenset({"a", "button", "img", "label", "option"})


@dataclass(frozen=True, slots=True)
class Extracted:
    title: str
    text: str
    links: list[str]  # уже канонічні, без дублікатів, у порядку появи


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.base: str | None = None
        self.hrefs: list[str] = []
        self.title: list[str] = []
        self.text: list[str] = []
        self._skip = 0
        self._in_title = False
        self.nofollow = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP:
            self._skip += 1
        elif tag == "base" and a.get("href"):
            self.base = a["href"]
        elif tag == "a" and a.get("href") and "nofollow" not in (a.get("rel") or ""):
            self.hrefs.append(a["href"] or "")
        elif tag == "meta" and (a.get("name") or "").lower() == "robots":
            self.nofollow = "nofollow" in (a.get("content") or "").lower()
        if tag in _BLOCK:
            self.text.append("\n")
        elif tag in _GAP:
            self.text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP and self._skip:
            self._skip -= 1
        if tag in _BLOCK:
            self.text.append("\n")
        elif tag in _GAP:
            self.text.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.text.append(data)


def extract(html: str, url: str) -> Extracted:
    """Заголовок, видимий текст і посилання сторінки (stdlib html.parser).

    Чиста CPU-робота: краулер викликає її через asyncio.to_thread, щоб велика
    сторінка не зупинила цикл подій.
    """
    p = _Parser()
    p.feed(html)
    p.close()
    base = canonicalize(p.base, url) if p.base else url
    links: list[str] = []
    seen: set[str] = set()
    if not p.nofollow:
        for href in p.hrefs:
            link = canonicalize(href, base or url)
            if link and link not in seen:
                seen.add(link)
                links.append(link)
    lines = (" ".join(line.split()) for line in "".join(p.text).splitlines())
    text = "\n".join(line for line in lines if line)
    return Extracted(" ".join("".join(p.title).split()), text, links)
