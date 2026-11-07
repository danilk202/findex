import posixpath
import re
from typing import Final
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_PORTS: Final = {"http": 80, "https": 443}
_PCT: Final = re.compile(r"%[0-9a-fA-F]{2}")


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Одна адреса — один рядок. None, якщо це не http(s) (mailto:, javascript:, ...).

    - схема і хост у нижньому регістрі, порт за замовчуванням прибирається;
    - фрагмент (#...) відкидається: це та сама сторінка;
    - шлях: порожній -> "/", подвійні слеші -> один, "." і ".." розкриваються;
      кінцевий слеш НЕ прибирається: /docs і /docs/ — різні ресурси, і сервери
      редіректять між ними; якби ми їх склеїли, редірект вів би в уже «бачену»
      адресу і сторінка загубилась би;
    - query: параметри сортуються (a=1&b=2 == b=2&a=1), порожній "?" зникає;
    - %-екранування у верхньому регістрі (%2f -> %2F).
    """
    if base is not None:
        url = urljoin(base, url.strip())
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:  # кривий порт, [ipv6 без дужки тощо
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None

    host = parts.hostname.lower()  # hostname уже без userinfo і в нижньому регістрі
    if ":" in host:  # IPv6
        host = f"[{host}]"
    netloc = host if port in (None, DEFAULT_PORTS[scheme]) else f"{host}:{port}"

    path = _upper_pct(parts.path) or "/"
    path = re.sub(r"/{2,}", "/", path)
    if "/." in path:
        trailing = path.endswith(("/", "/.", "/.."))
        path = posixpath.normpath(path)
        if path.startswith("//"):  # normpath лишає два слеші на початку (POSIX)
            path = "/" + path.lstrip("/")
        if trailing and path != "/":
            path += "/"

    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, _upper_pct(query), ""))


def _upper_pct(s: str) -> str:
    return _PCT.sub(lambda m: m.group().upper(), s)


def host_key(url: str) -> str:
    """Ключ «хоста» для лімітів: host:port (різні порти — різні сервери)."""
    return urlsplit(url).netloc


def origin(url: str) -> str:
    """scheme://host:port — область дії robots.txt (RFC 9309)."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"
