"""End-to-end check without an LLM: hand-written answers must reproduce the
assignment's expected output byte for byte.

FAKE_ANSWERS is what we expect a good model to return yet: for each <p>, a list
of (first token, last token, type, correction). Token numbers come from tokenize().
"""

from pathlib import Path

from proofread.injector import annotate_paragraph
from proofread.models import ErrorSpan
from proofread.paragraph import paragraph_text, text_slots
from proofread.tokenizer import token_range_to_chars, tokenize
from proofread.xml_io import find_paragraphs, load_xml, serialize

SAMPLES = Path(__file__).parent.parent / "samples"

FAKE_ANSWERS = [
    # For breakfast I like eggs toast and juice.
    [
        (3, 3, "punctuation", ", "),  # " " after "breakfast"
        (9, 9, "punctuation", ", "),  # " " after "eggs"
        (11, 12, "punctuation", ", and"),  # " and"
    ],
    # The companys employees work hard but their unhappy.
    [
        (2, 2, "punctuation", "company's"),
        (9, 9, "punctuation", ", "),
        (12, 12, "grammar", "they're"),
    ],
    # They always eat dinner together at home.
    [],
    # sarah lives in paris france? She enjoy french food.
    [
        (0, 0, "capitalization", "Sarah"),
        (6, 8, "capitalization", "Paris, France"),  # "paris france"
        (9, 9, "punctuation", "."),  # "?"
        (13, 13, "grammar", "enjoys"),
    ],
    # i visited london last summer and i loved the museums the galleries and the food.
    [
        (0, 0, "capitalization", "I"),
        (4, 4, "capitalization", "London"),
        (12, 12, "capitalization", "I"),
        (19, 19, "punctuation", ", "),
        (23, 23, "punctuation", ", "),
    ],
]


def to_spans(p, answers) -> list[ErrorSpan]:
    """Convert token-number answers into character-position ErrorSpans."""
    tokens = tokenize([slot.value for slot in text_slots(p)])
    spans = []
    for first, last, error_type, correction in answers:
        start, end = token_range_to_chars(tokens, first, last)
        spans.append(ErrorSpan(start, end, error_type, correction))
    return spans


def test_reproduces_the_assignment_output_exactly():
    doc = load_xml(SAMPLES / "input.xml")
    paragraphs = find_paragraphs(doc.root)
    assert len(paragraphs) == len(FAKE_ANSWERS)

    for p, answers in zip(paragraphs, FAKE_ANSWERS, strict=True):
        before = paragraph_text(p)
        result = annotate_paragraph(p, to_spans(p, answers))
        assert result.rejected == []
        assert len(result.applied) == len(answers)
        assert paragraph_text(p) == before  # the length invariant

    assert serialize(doc) == (SAMPLES / "expected_output.xml").read_bytes()
