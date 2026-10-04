from findex.tokenize import tokenize


def t(s):
    return list(tokenize(s))


def test_case():
    assert t("Hello WORLD") == ["hello", "world"]


def test_cyrillic():
    assert t("Привіт світ") == ["привіт", "світ"]


def test_apostrophe():
    assert t("п'ять don't") == ["п'ять", "don't"]


def test_hyphen():
    assert t("well-known") == ["well-known"]


def test_casefold():
    assert t("Straße") == ["strasse"]


def test_punctuation():
    assert t("a, b; c!") == ["a", "b", "c"]


def test_digits():
    assert t("mp3 2026") == ["mp3", "2026"]


def test_edge_quotes():
    assert t("'hello'") == ["hello"]


def test_empty():
    assert t("") == []


def test_combining_mark():

    assert t("cafe\u0301") == t("caf\u00e9") == ["caf\u00e9"]
