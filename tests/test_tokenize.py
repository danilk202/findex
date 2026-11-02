import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello WORLD", ["hello", "world"]),
        ("a, b; c!", ["a", "b", "c"]),
        ("", []),
        ("   \n\t ", []),
        ("mp3 2026", ["mp3", "2026"]),
        ("snake_case", ["snake_case"]),
        # апостроф і дефіс усередині слова — частина токена, на краях — ні
        ("don't", ["don't"]),
        ("'hello'", ["hello"]),
        ("well-known", ["well-known"]),
        ("-dash-", ["dash"]),
        # Unicode
        ("Привіт світ", ["привіт", "світ"]),
        ("п'ять", ["п'ять"]),
        ("пʼять", ["пʼять"]),  # U+02BC — «український» апостроф
        ("п’ять", ["п’ять"]),  # U+2019 — типографський
        ("ЇЖАК Ґанок", ["їжак", "ґанок"]),
        ("Straße", ["strasse"]),  # casefold, а не lower
        ("ΣΟΦΙΑ", ["σοφια"]),
        ("café", ["café"]),  # NFC: e + акут == é
        ("naïve café", ["naïve", "café"]),
        ("日本語 テキスト", ["日本語", "テキスト"]),
        ("emoji 🐋 whale", ["emoji", "whale"]),
    ],
)
def test_tokenize(text: str, expected: list[str]):
    assert list(tokenize(text)) == expected


def test_composed_and_decomposed_are_equal():
    assert list(tokenize("café")) == list(tokenize("café"))


def test_is_lazy():
    gen = tokenize("a b c")
    assert next(gen) == "a"  # генератор, а не список
