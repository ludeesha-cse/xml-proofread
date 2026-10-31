"""Tests for injector.py."""

import pytest
from lxml import etree

from proofread.injector import validate_spans
from proofread.models import ErrorSpan
from proofread.paragraph import paragraph_text, text_slots


def first_p(xml: str) -> etree._Element:
    return next(etree.fromstring(xml.encode()).iter("{*}p"))


def span(start: int, end: int, correction: str = "X", type: str = "grammar") -> ErrorSpan:
    return ErrorSpan(start=start, end=end, type=type, correction=correction)


# --- validate_spans --------------------------------------------------------------


def check(xml: str, spans: list[ErrorSpan]):
    p = first_p(xml)
    return validate_spans(spans, paragraph_text(p), text_slots(p))


def test_valid_spans_are_accepted_and_sorted():
    accepted, rejected = check("<r><p>a b c</p></r>", [span(4, 5), span(0, 1)])
    assert [(s.start, s.end) for s in accepted] == [(0, 1), (4, 5)]
    assert rejected == []


# Text "a b cde": p.text is "a b " (0-4), <b>'s text is "cde" (4-7).
@pytest.mark.parametrize(
    ("spans", "reason"),
    [
        ([span(0, 1, type="style")], "unknown type 'style'"),
        ([span(5, 99)], "invalid position 5-99"),
        ([span(2, 2)], "invalid position 2-2"),  # empty
        ([span(0, 1, correction="a")], "correction is the same as the original"),
        ([span(2, 5)], "crosses a tag boundary"),  # "b c" starts in <p>, ends in <b>
        ([span(0, 3), span(2, 3)], "overlaps another error"),
    ],
)
def test_rejection_reasons(spans, reason):
    _, rejected = check("<r><p>a b <b>cde</b></p></r>", spans)
    assert [r for _, r in rejected] == [reason]


def test_adjacent_spans_do_not_overlap():
    accepted, rejected = check("<r><p>paris france?</p></r>", [span(0, 12), span(12, 13)])
    assert len(accepted) == 2 and rejected == []
