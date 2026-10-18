import random

import pytest

from findex.boolean import and_merge, and_set, evaluate, or_merge, or_set
from findex.models import Posting


@pytest.fixture
def postings():
    def mk(*ids):
        return [Posting(i, 1) for i in ids]

    return {"cat": mk(0, 1, 3), "dog": mk(1, 2), "fish": mk(3, 4)}


@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("cat dog", [1]),
        ("cat OR dog", [0, 1, 2, 3]),
        ("cat NOT dog", [0, 3]),
        ("NOT cat", [2, 4]),
        ("cat zebra", []),
        ("zebra OR fish", [3, 4]),
        ("CAT", [0, 1, 3]),
    ],
)
def test_queries(postings, engine, query, expected):
    assert evaluate(query, postings, 5, engine) == expected


def test_merge_and_set_agree_on_random_lists():
    rng = random.Random(42)
    for _ in range(200):
        a = sorted(rng.sample(range(100), rng.randint(0, 30)))
        b = sorted(rng.sample(range(100), rng.randint(0, 30)))
        assert and_merge(a, b) == and_set(a, b)
        assert or_merge(a, b) == or_set(a, b)