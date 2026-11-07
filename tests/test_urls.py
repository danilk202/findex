import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.urls import canonicalize, host_key, origin


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Example.COM/a", "http://example.com/a"),  # схема і хост — нижній регістр
        ("http://example.com/A", "http://example.com/A"),  # шлях — ні: він чутливий до регістру
        ("http://example.com", "http://example.com/"),  # порожній шлях
        ("http://example.com:80/x", "http://example.com/x"),  # порт за замовчуванням
        ("https://example.com:443/x", "https://example.com/x"),
        ("http://example.com:8080/x", "http://example.com:8080/x"),
        ("http://example.com/x#section-2", "http://example.com/x"),  # фрагмент геть
        ("http://example.com/x?b=2&a=1", "http://example.com/x?a=1&b=2"),  # порядок query
        ("http://example.com/x?", "http://example.com/x"),  # порожній query
        ("http://example.com//a///b", "http://example.com/a/b"),  # подвійні слеші
        ("http://example.com/a/./b/../c", "http://example.com/a/c"),  # . і ..
        ("http://example.com/a/b/..", "http://example.com/a/"),
        ("http://example.com/docs/", "http://example.com/docs/"),  # кінцевий слеш лишається
        ("http://example.com/%7euser/%2f", "http://example.com/%7Euser/%2F"),
        ("http://user:pw@Example.com/", "http://example.com/"),  # без облікових даних
        ("  http://example.com/x  ", "http://example.com/x"),
    ],
)
def test_canonicalize(raw: str, expected: str):
    assert canonicalize(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["mailto:a@b.c", "javascript:void(0)", "ftp://example.com/", "data:text/plain,hi", "", "/x"],
)
def test_not_http_is_none(raw: str):
    assert canonicalize(raw) is None


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("b", "http://h.com/dir/b"),
        ("../up", "http://h.com/up"),
        ("/abs?z=1&y=2#f", "http://h.com/abs?y=2&z=1"),
        ("//other.org/p", "http://other.org/p"),
        ("#only-fragment", "http://h.com/dir/page"),
    ],
)
def test_relative_links(href: str, expected: str):
    assert canonicalize(href, "http://h.com/dir/page") == expected


segments = st.lists(st.sampled_from(["a", "B", ".", "..", "", "x y", "%41"]), max_size=5)
queries = st.lists(st.tuples(st.sampled_from("abc"), st.sampled_from(["1", "", "x y"])), max_size=4)


@given(
    scheme=st.sampled_from(["http", "HTTP", "https"]),
    host=st.sampled_from(["Example.com", "h.org", "127.0.0.1"]),
    port=st.sampled_from(["", ":80", ":443", ":8100"]),
    path=segments,
    query=queries,
    frag=st.sampled_from(["", "#x", "#"]),
)
def test_canonical_is_idempotent(
    scheme: str, host: str, port: str, path: list[str], query: list[tuple[str, str]], frag: str
):
    q = "&".join(f"{k}={v}" for k, v in query)
    url = f"{scheme}://{host}{port}/{'/'.join(path)}{'?' + q if q else ''}{frag}"
    once = canonicalize(url)
    assert once is not None
    assert canonicalize(once) == once  # канонічна форма канонічна
    assert "#" not in once


def test_host_key_and_origin():
    assert host_key("http://127.0.0.1:8100/a") == "127.0.0.1:8100"
    assert origin("https://example.com/a/b?c") == "https://example.com"
