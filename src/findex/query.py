"""Мова запитів: python AND (async OR await) NOT java "event loop"

Граматика (від найслабшого оператора до найсильнішого):

    or_expr  := and_expr ("OR" and_expr)*
    and_expr := not_expr (["AND"] not_expr)*     # AND можна не писати
    not_expr := "NOT" not_expr | atom
    atom     := WORD | PHRASE | "(" or_expr ")"

Кожне правило — окрема функція, і вони викликають одна одну: це рекурсивний спуск.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from findex.index import Index
from findex.tokenize import tokenize


class QuerySyntaxError(ValueError):
    """Запит не відповідає граматиці."""


class NeedsPositionsError(ValueError):
    """Фразовий запит до індексу, зібраного без позицій."""


# ---------------------------------------------------------------- вузли дерева


class Node:
    """Спільна база: оператори &, |, ~ будують дерево прямо в Python-коді."""

    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def terms(self) -> Iterator[str]:
        """Терми для ранжування — все, крім того, що під NOT."""
        return iter(())

    # __str__ — для людей і логів; __repr__ (від dataclass) — для налагодження й тестів


@dataclass(frozen=True)
class Term(Node):
    text: str

    def __str__(self) -> str:
        return self.text

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_ids(self.text).tolist())

    def terms(self) -> Iterator[str]:
        yield self.text


@dataclass(frozen=True)
class Phrase(Node):
    words: tuple[str, ...]

    def __str__(self) -> str:
        return '"' + " ".join(self.words) + '"'

    def evaluate(self, index: Index) -> set[int]:
        if not index.has_positions:
            raise NeedsPositionsError("phrase search needs an index built with positions")
        arr = index.arrays
        docs: list[set[int]] = [set(arr.doc_ids_of(w).tolist()) for w in self.words]
        if not all(docs):
            return set()
        found: set[int] = set()
        for doc in docs[0].intersection(*docs[1:]):
            # початки фрази: позиції першого слова, для яких слово i стоїть на start + i
            starts = arr.positions_in(self.words[0], doc)
            for i, w in enumerate(self.words[1:], 1):
                starts = starts[np.isin(starts + i, arr.positions_in(w, doc))]
                if not len(starts):
                    break
            if len(starts):
                found.add(doc)
        return found

    def terms(self) -> Iterator[str]:
        yield from self.words


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def __str__(self) -> str:
        return f"NOT {self.child}"

    def evaluate(self, index: Index) -> set[int]:
        return set(range(index.num_docs)) - self.child.evaluate(index)


# ---------------------------------------------------------------- лексер

_TOKEN = re.compile(r'\s*(?:(?P<phrase>"[^"]*")|(?P<op>[()])|(?P<word>[^\s()"]+))')
KEYWORDS = {"AND", "OR", "NOT"}


def lex(query: str) -> list[str]:
    """'a AND (b "c d")' -> ['a', 'AND', '(', 'b', '"c d"', ')']"""
    tokens: list[str] = []
    pos = 0
    query = query.rstrip()
    while pos < len(query):
        m = _TOKEN.match(query, pos)
        if not m:
            raise QuerySyntaxError(f"unclosed quote at position {pos}")
        assert m.lastgroup is not None  # регулярка завжди збігається з однією з трьох груп
        tokens.append(m.group(m.lastgroup))
        pos = m.end()
    return tokens


# ---------------------------------------------------------------- парсер


class Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("unexpected end of query")
        self.i += 1
        return tok

    # or_expr := and_expr ("OR" and_expr)*
    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = node | self.and_expr()
        return node

    # and_expr := not_expr (["AND"] not_expr)*
    def and_expr(self) -> Node:
        node = self.not_expr()
        while self.peek() not in (None, "OR", ")"):
            if self.peek() == "AND":
                self.take()
            node = node & self.not_expr()
        return node

    # not_expr := "NOT" not_expr | atom
    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return ~self.not_expr()
        return self.atom()

    # atom := WORD | PHRASE | "(" or_expr ")"
    def atom(self) -> Node:
        tok = self.take()
        if tok == "(":
            node = self.or_expr()
            if self.take() != ")":
                raise QuerySyntaxError("expected ')'")
            return node
        if tok in KEYWORDS or tok == ")":
            raise QuerySyntaxError(f"unexpected {tok!r}")
        words = list(tokenize(tok.strip('"')))
        if not words:
            raise QuerySyntaxError(f"empty term {tok!r}")
        # «don't» — один токен, а "event-loop!" чи "event loop" — кілька: тоді це фраза
        return Phrase(tuple(words)) if len(words) > 1 else Term(words[0])


def parse(query: str) -> Node:
    parser = Parser(lex(query))
    if parser.peek() is None:
        raise QuerySyntaxError("empty query")
    node = parser.or_expr()
    if parser.peek() is not None:
        raise QuerySyntaxError(f"unexpected {parser.peek()!r}")
    return node
