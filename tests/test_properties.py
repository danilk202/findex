"""Властивості (Hypothesis): не конкретні приклади, а твердження «для будь-яких даних».

Hypothesis генерує сотні входів, а коли знаходить контрприклад — зменшує його
до мінімального (shrinking), щоб помилку було легко зрозуміти.
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.boolean import and_merge, andnot_merge, or_merge
from findex.corpus import Document
from findex.index import Index
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.scoring import BM25, TfIdf

# --- стратегії -------------------------------------------------------------

WORDS = ["whale", "ship", "sea", "captain", "ahab", "white", "the", "a"]

texts = st.lists(st.sampled_from(WORDS), max_size=40).map(" ".join)
corpora = st.lists(texts, min_size=1, max_size=8)
sorted_ids = st.lists(st.integers(0, 200), unique=True, max_size=60).map(sorted)


def build(docs: list[str], positions: bool) -> Index:
    return Index.build(
        (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(docs)),
        with_positions=positions,
    )


# --- 1. load(save(index)) == index --------------------------------------------


@pytest.mark.slow
@settings(max_examples=1000, deadline=None)
@given(corpora, st.booleans(), st.sampled_from(["pickle", "json"]))
def test_save_load_roundtrip(docs: list[str], positions: bool, fmt):
    index = build(docs, positions)
    # tmp_path — function-scoped фікстура, а Hypothesis запускає тест сотні разів
    # в одному виклику, тому тимчасовий каталог створюємо самі
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx"
        index.save(path, fmt)
        loaded = Index.load(path)
    assert loaded == index
    assert loaded.metas == index.metas


# --- 2. постінги завжди відсортовані й узгоджені ------------------------------


@given(corpora, st.booleans())
def test_postings_invariants(docs: list[str], positions: bool):
    index = build(docs, positions)
    for term, plist in index.items():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(set(ids)), term  # строго зростають, без дублікатів
        for p in plist:
            assert p.tf >= 1
            if positions:
                assert p.positions is not None
                assert len(p.positions) == p.tf
                assert list(p.positions) == sorted(p.positions)
    assert index.num_tokens == sum(len(d.split()) for d in docs)


# --- 3. merge збігається з операціями над set ---------------------------------


@given(sorted_ids, sorted_ids)
def test_merge_matches_set_ops(a: list[int], b: list[int]):
    assert and_merge(a, b) == sorted(set(a) & set(b))
    assert or_merge(a, b) == sorted(set(a) | set(b))
    assert andnot_merge(a, b) == sorted(set(a) - set(b))


# --- 4. parse(str(tree)) == tree --------------------------------------------------

terms = st.sampled_from(WORDS).map(Term)
phrases = st.lists(st.sampled_from(WORDS), min_size=2, max_size=3).map(lambda ws: Phrase(tuple(ws)))
trees: st.SearchStrategy[Node] = st.recursive(
    terms | phrases,
    lambda children: st.one_of(
        st.builds(And, children, children),
        st.builds(Or, children, children),
        st.builds(Not, children),
    ),
    max_leaves=12,
)


@given(trees)
def test_parse_str_roundtrip(tree: Node):
    assert parse(str(tree)) == tree


# --- 5. BM25: монотонний по tf і обмежений стелею ------------------------------


class Stats:
    num_docs = 25
    avg_doc_length = 1000.0


@given(
    st.integers(1, 10_000),
    st.integers(1, 25),
    st.integers(1, 100_000),
)
def test_bm25_monotonic_and_bounded(tf: int, df: int, doc_len: int):
    s = BM25()
    score, more = s.score(tf, df, doc_len, Stats()), s.score(tf + 1, df, doc_len, Stats())
    ceiling = s.score(10**12, df, doc_len, Stats())
    assert 0 < score <= more <= ceiling * (1 + 1e-9)
    # TF-IDF без стелі: лінійний по tf
    t = TfIdf()
    assert t.score(2 * tf, df, doc_len, Stats()) == pytest.approx(
        2 * t.score(tf, df, doc_len, Stats())
    )
