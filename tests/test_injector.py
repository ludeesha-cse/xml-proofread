"""Tests for injector.py."""

import pytest
from lxml import etree

from proofread import injector
from proofread.injector import annotate_paragraph, validate_spans
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


# --- annotate_paragraph --------------------------------------------------------------


def annotate(xml: str, spans: list[ErrorSpan]) -> tuple[str, etree._Element]:
    """Annotate the first <p>; return the <p> as a string, plus the root element.

    Also checks the length invariant on every call, so every test below verifies it.
    """
    root = etree.fromstring(xml.encode())
    p = next(root.iter("{*}p"))
    before = paragraph_text(p)
    annotate_paragraph(p, spans)
    assert paragraph_text(p) == before
    return etree.tostring(p, with_tail=False).decode(), root


def test_wrap_one_word():
    out, _ = annotate("<r><p>My name is john.</p></r>", [span(11, 15, "John", "capitalization")])
    assert out == '<p>My name is <error type="capitalization" correction="John">john</error>.</p>'


def test_wrap_a_space_for_a_missing_comma():
    out, _ = annotate("<r><p>eggs toast</p></r>", [span(4, 5, ", ", "punctuation")])
    assert out == '<p>eggs<error type="punctuation" correction=", "> </error>toast</p>'


def test_several_errors_in_one_slot():
    out, _ = annotate("<r><p>a b c</p></r>", [span(4, 5, "C"), span(0, 1, "A")])
    assert out == (
        '<p><error type="grammar" correction="A">a</error> b '
        '<error type="grammar" correction="C">c</error></p>'
    )


def test_error_at_the_very_end():
    out, _ = annotate("<r><p>ok?</p></r>", [span(2, 3, ".", "punctuation")])
    assert out == '<p>ok<error type="punctuation" correction=".">?</error></p>'


def test_errors_inside_and_after_an_inline_tag():
    # "Hello big world": "big" is in <b>, "world" is in <b>'s tail.
    out, _ = annotate(
        '<r><p id="1">Hello <b>big</b> world</p></r>', [span(6, 9, "BIG"), span(10, 15, "World")]
    )
    assert out == (
        '<p id="1">Hello <b><error type="grammar" correction="BIG">big</error></b> '
        '<error type="grammar" correction="World">world</error></p>'
    )


def test_error_after_a_comment():
    out, _ = annotate("<r><p>a<!--c--> b</p></r>", [span(2, 3, "B")])
    assert out == '<p>a<!--c--> <error type="grammar" correction="B">b</error></p>'


def test_default_namespace_writes_plain_error_tag():
    _, root = annotate('<r xmlns="http://x"><p>a</p></r>', [span(0, 1, "A")])
    assert etree.tostring(root).decode() == (
        '<r xmlns="http://x"><p><error type="grammar" correction="A">a</error></p></r>'
    )
    assert root[0][0].tag == "{http://x}error"  # same namespace as its neighbours


def test_prefixed_namespace_writes_plain_error_tag():
    out, _ = annotate('<r xmlns:h="http://h"><h:p>a</h:p></r>', [span(0, 1, "A")])
    assert '<error type="grammar" correction="A">a</error>' in out


def test_nested_paragraph_is_untouched():
    out, _ = annotate("<r><p>one <p>two</p> three</p></r>", [span(4, 5, "T")])
    # Outer text is "one  three"; position 4-5 is the space before "three".
    assert out == '<p>one <p>two</p><error type="grammar" correction="T"> </error>three</p>'


def test_invalid_spans_change_nothing():
    xml = "<r><p>a b <b>cde</b></p></r>"
    out, _ = annotate(xml, [span(2, 5), span(0, 99)])
    assert out == "<p>a b <b>cde</b></p>"


def test_cdata_paragraph_is_left_alone():
    root = etree.fromstring(b"<r><p>x <![CDATA[a<b]]></p></r>", etree.XMLParser(strip_cdata=False))
    result = annotate_paragraph(root[0], [span(0, 1)])
    assert result.applied == []
    assert result.rejected[0][1] == "paragraph contains CDATA"
    assert b"<![CDATA[a<b]]>" in etree.tostring(root)


def test_result_lists_applied_and_rejected():
    root = etree.fromstring(b"<r><p>a b</p></r>")
    result = annotate_paragraph(root[0], [span(0, 1), span(0, 99)])
    assert len(result.applied) == 1
    assert len(result.rejected) == 1


def test_reason_is_written_when_present():
    reasoned = ErrorSpan(0, 1, "grammar", "A", reason="Sentences start with a capital.")
    out, _ = annotate("<r><p>a</p></r>", [reasoned])
    assert 'reason="Sentences start with a capital."' in out


def test_no_reason_attribute_without_a_reason():
    out, _ = annotate("<r><p>a</p></r>", [ErrorSpan(0, 1, "grammar", "A")])
    assert "reason=" not in out


# --- safety net ---------------------------------------------------------------------


def test_text_check_undoes_a_buggy_injection(monkeypatch):
    """Simulate a bug that corrupts the wrapped text; the paragraph must come back intact."""
    real_make_error = injector._make_error

    def buggy_make_error(parent, span, original):
        return real_make_error(parent, span, original.upper())  # changes the text!

    monkeypatch.setattr(injector, "_make_error", buggy_make_error)

    xml = '<r><p id="1">Hello <b>big</b> wide world, a b c.</p></r>'
    root = etree.fromstring(xml.encode())
    result = annotate_paragraph(root[0], [span(0, 5), span(6, 9), span(10, 14), span(25, 26)])

    assert result.applied == []
    assert {reason for _, reason in result.rejected} == {"failed the text check"}
    assert etree.tostring(root).decode() == xml  # byte-identical to before
