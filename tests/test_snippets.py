from bisect import bisect_left, bisect_right

from hypothesis import given
from hypothesis import strategies as st

from findex.snippets import best_window, find_hits, make_snippet
from findex.tokenize import WORD, normalize

FILLER = "lorem ipsum dolor sit amet " * 20  # ~540 символів «нічого»


def test_highlights_terms_case_insensitive():
    s = make_snippet("Call me Ishmael. The Whale was white.", ["whale"])
    assert "**Whale**" in s


def test_window_size():
    text = FILLER + "the whale swims " + FILLER
    s = make_snippet(text, ["whale"], width=80)
    assert "**whale**" in s
    assert s.startswith("… ") and s.endswith(" …")
    assert len(s) <= 2 * 80 + len("**whale**") + 10


def test_does_not_cut_words():
    text = FILLER + "the whale swims " + FILLER
    inner = make_snippet(text, ["whale"]).strip("… ")
    words = set(inner.replace("**", "").split())
    assert words <= {"lorem", "ipsum", "dolor", "sit", "amet", "the", "whale", "swims"}


def test_picks_window_with_most_distinct_terms():
    # перше входження whale самотнє, друге — поруч із ship і captain
    text = "whale " + FILLER + "the captain saw a whale from the ship " + FILLER
    s = make_snippet(text, ["whale", "ship", "captain"])
    assert "**captain**" in s and "**ship**" in s


def test_no_hit_returns_beginning():
    s = make_snippet("first words of the book " + FILLER, ["zebra"])
    assert s.startswith("first words")


def test_collapses_newlines():
    s = make_snippet("the white\r\n\r\nwhale\nswims", ["whale"])
    assert s == "the white **whale** swims"


def test_term_at_window_edge_is_not_cut():
    # captain стоїть рівно на межі вікна навколо whale — його не можна обрізати
    text = FILLER + "whale" + " " * 74 + "captain " + FILLER
    s = make_snippet(text, ["whale", "captain"], width=80)
    assert "**whale**" in s and "**captain**" in s


# --- лаба 7: швидкий пошук входжень і рання зупинка не змінюють результату ----

VOCAB = ["whale", "Whale", "WHALE", "ship", "sea", "don't", "well-known", "known", "the", "a"]
SEPS = [" ", " ", "\n", ", ", "-", "'"]
texts = st.lists(st.tuples(st.sampled_from(VOCAB), st.sampled_from(SEPS)), max_size=80).map(
    lambda xs: "".join(w + s for w, s in xs)
)
queries = st.lists(
    st.sampled_from(["whale", "ship", "sea", "known", "don't", "the"]), min_size=1, max_size=3
).map(set)


def full_scan(text: str, wanted: set[str]) -> list[tuple[int, int, str]]:
    # як було в лабі 3: токенізуємо весь текст
    words = ((m.start(), m.end(), normalize(m.group())) for m in WORD.finditer(text))
    return [h for h in words if h[2] in wanted]


def reference_best(text: str, wanted: set[str], width: int) -> int | None:
    # вибір вікна з лаби 3: перебір усіх входжень, без ранньої зупинки
    hits = full_scan(text, wanted)
    if not hits:
        return None
    starts = [s for s, _, _ in hits]
    ends = [e for _, e, _ in hits]

    def distinct(i: int) -> int:
        lo = bisect_left(starts, hits[i][0] - width)
        hi = bisect_right(ends, hits[i][1] + width)
        return len({w for _, _, w in hits[lo:hi]})

    return hits[max(range(len(hits)), key=lambda i: (distinct(i), -i))][0]


@given(texts, queries)
def test_find_hits_equals_full_tokenization(text: str, wanted: set[str]):
    assert find_hits(text, wanted) == full_scan(text, wanted)


@given(texts, queries, st.sampled_from([5, 20, 80]))
def test_early_stop_picks_same_window(text: str, wanted: set[str], width: int):
    hits, best = best_window(text, wanted, width)
    got = hits[best][0] if best is not None else None
    assert got == reference_best(text, wanted, width)
