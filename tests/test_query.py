import pytest

from findex.corpus import iter_documents
from findex.index import Index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, lex, parse

a, b, c, d = Term("a"), Term("b"), Term("c"), Term("d")


# ---------- лексер ----------


def test_lex():
    assert lex('a AND (b "c d")') == ["a", "AND", "(", "b", '"c d"', ")"]
    assert lex("(a)") == ["(", "a", ")"]


# ---------- оператори &, |, ~ ----------


def test_operators_build_nodes():
    assert (a & b) == And(a, b)
    assert (a | b) == Or(a, b)
    assert ~a == Not(a)
    assert (a & ~b | c) == Or(And(a, Not(b)), c)  # пріоритет як у Python: ~ > & > |


# ---------- парсер: порівнюємо дерева, не рядки ----------


@pytest.mark.parametrize(
    ("query", "tree"),
    [
        ("a", a),
        ("a AND b", a & b),
        ("a b", a & b),  # неявний AND
        ("a OR b", a | b),
        ("a OR b c", a | (b & c)),  # AND сильніший за OR
        ("a b OR c", (a & b) | c),
        ("(a OR b) c", (a | b) & c),  # дужки змінюють пріоритет
        ("NOT a", ~a),
        ("a NOT b", a & ~b),
        ("NOT NOT a", ~~a),
        ("a OR b OR c", (a | b) | c),  # ліва асоціативність
        ("A or B", a & Term("or") & b),  # оператори лише ВЕЛИКИМИ
        ('"a b"', Phrase(("a", "b"))),
        ('"a"', a),  # фраза з одного слова = терм
        (
            'python AND (async OR await) NOT java "event loop"',
            Term("python")
            & (Term("async") | Term("await"))
            & ~Term("java")
            & Phrase(("event", "loop")),
        ),
    ],
)
def test_parse(query, tree):
    assert parse(query) == tree


@pytest.mark.parametrize("bad", ["", "(a", "a)", "a AND", "OR a", "NOT", '"a b', "()", '""'])
def test_syntax_errors(bad):
    with pytest.raises(QuerySyntaxError):
        parse(bad)


def test_terms_skip_not():
    assert list(parse('a NOT b "c d"').terms()) == ["a", "c", "d"]


# ---------- evaluate на маленькому корпусі ----------


@pytest.fixture
def index(tmp_path):
    texts = {
        "d0": "python async await event loop",
        "d1": "python java loop event",
        "d2": "java coffee",
        "d3": "python event loop",
    }
    for name, text in texts.items():
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    return Index.build(iter_documents(tmp_path), with_positions=True)


@pytest.mark.parametrize(
    ("query", "ids"),
    [
        ("python", {0, 1, 3}),
        ("python java", {1}),
        ("python OR coffee", {0, 1, 2, 3}),
        ("python NOT java", {0, 3}),
        ("NOT python", {2}),
        ('"event loop"', {0, 3}),  # у d1 "loop event" — не фраза
        ('python AND (async OR await) NOT java "event loop"', {0}),
        ("zebra", set()),
    ],
)
def test_evaluate(index, query, ids):
    assert parse(query).evaluate(index) == ids


def test_phrase_needs_positions(tmp_path):
    (tmp_path / "x.txt").write_text("event loop", encoding="utf-8")
    idx = Index.build(iter_documents(tmp_path))
    with pytest.raises(ValueError, match="positions"):
        parse('"event loop"').evaluate(idx)


def test_str_is_readable():
    assert str(parse('python AND (async OR await) NOT java "event loop"')) == (
        '(((python AND (async OR await)) AND NOT java) AND "event loop")'
    )


def test_nodes_are_hashable():
    # frozen dataclass -> є __hash__, тож дерево може бути ключем кешу
    assert hash(parse("a b")) == hash(parse("a AND b"))
