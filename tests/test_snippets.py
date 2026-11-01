from findex.snippets import make_snippet

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
