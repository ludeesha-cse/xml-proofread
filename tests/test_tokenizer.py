"""Tests for tokenizer.py."""

from itertools import pairwise

import pytest

from proofread.tokenizer import token_range_to_chars, tokenize


def texts(pieces: list[str]) -> list[str]:
    return [t.text for t in tokenize(pieces)]


def test_words_spaces_and_punctuation():
    assert texts(["eggs toast, and juice."]) == [
        "eggs", " ", "toast", ",", " ", "and", " ", "juice", ".",
    ]  # fmt: skip


def test_apostrophes_stay_inside_words():
    assert texts(["they're company's l’école"]) == [
        "they're", " ", "company's", " ", "l’école",
    ]  # fmt: skip


def test_whitespace_runs_are_one_token():
    assert texts(["a \n\t b"]) == ["a", " \n\t ", "b"]


def test_unicode_letters_are_word_characters():
    assert texts(["élève Ça 日本"]) == ["élève", " ", "Ça", " ", "日本"]


def test_positions_count_across_pieces():
    tokens = tokenize(["Hello ", "big", " world"])
    assert [(t.text, t.start, t.end) for t in tokens] == [
        ("Hello", 0, 5), (" ", 5, 6), ("big", 6, 9), (" ", 9, 10), ("world", 10, 15),
    ]  # fmt: skip


def test_tokens_never_cross_pieces():
    # "Hel" + "lo" from <p>Hel<b>lo</b></p> stay two tokens.
    assert texts(["Hel", "lo"]) == ["Hel", "lo"]


@pytest.mark.parametrize(
    "text", ["", "  ", "Hello, world!", "a--b...c", "«Bonjour» dit-il.", "x_y 3.14 #tag"]
)
def test_tokens_rebuild_the_text_exactly(text):
    tokens = tokenize([text])
    assert "".join(t.text for t in tokens) == text
    # Positions are continuous: each token starts where the previous one ended.
    assert all(a.end == b.start for a, b in pairwise(tokens))


def test_token_range_to_chars():
    tokens = tokenize(["For breakfast I like eggs toast and juice."])
    assert token_range_to_chars(tokens, 11, 12) == (31, 35)  # " and"
    assert token_range_to_chars(tokens, 2, 2) == (4, 13)  # "breakfast"


@pytest.mark.parametrize(("first", "last"), [(-1, 0), (3, 2), (0, 99)])
def test_token_range_to_chars_rejects_bad_ranges(first, last):
    with pytest.raises(ValueError):
        token_range_to_chars(tokenize(["a b c"]), first, last)
