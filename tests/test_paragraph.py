"""Tests for paragraph.py."""

from lxml import etree

from proofread.paragraph import paragraph_text, text_slots

PARSER = etree.XMLParser(strip_cdata=False, resolve_entities=False)


def first_p(xml: str) -> etree._Element:
    """Parse xml and return its first <p>."""
    root = etree.fromstring(xml.encode(), PARSER)
    return next(root.iter("{*}p"))


def test_plain_paragraph_is_one_slot():
    p = first_p("<r><p>My name is john.</p></r>")
    slots = text_slots(p)
    assert [(s.attr, s.start, s.end, s.value) for s in slots] == [
        ("text", 0, 16, "My name is john.")
    ]


def test_mixed_content_slots():
    p = first_p("<r><p>Hello <b>big</b> world</p></r>")
    assert [(s.owner.tag, s.attr, s.start, s.end) for s in text_slots(p)] == [
        ("p", "text", 0, 6),
        ("b", "text", 6, 9),
        ("b", "tail", 9, 15),
    ]
    assert paragraph_text(p) == "Hello big world"


def test_deeply_nested_inline_elements():
    p = first_p("<r><p>a <b>b <i>c</i> d</b> e</p></r>")
    assert paragraph_text(p) == "a b c d e"


def test_nested_paragraph_content_is_excluded():
    p = first_p("<r><p>before <p>inner</p> after</p></r>")
    assert paragraph_text(p) == "before  after"


def test_comments_and_pis_are_excluded_but_their_tail_is_kept():
    p = first_p("<r><p>a<!-- note --> b<?pi x?> c</p></r>")
    assert paragraph_text(p) == "a b c"


def test_empty_paragraph_has_no_slots():
    assert text_slots(first_p("<r><p/></r>")) == []
    assert paragraph_text(first_p("<r><p><b/></p></r>")) == ""


def test_text_after_the_paragraph_is_not_included():
    p = first_p("<r><p>inside</p> outside</r>")
    assert paragraph_text(p) == "inside"