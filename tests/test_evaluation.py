"""Tests for evaluation.py."""

from pathlib import Path

import pytest
from lxml import etree

from proofread.evaluation import corrected_text, edits, score

SAMPLES = Path(__file__).parent.parent / "samples"


def doc(body: str) -> etree._Element:
    return etree.fromstring(f"<r>{body}</r>".encode())


def err(text: str, correction: str) -> str:
    return f'<error type="grammar" correction="{correction}">{text}</error>'


def test_corrected_text_applies_every_correction():
    p = doc(f"<p>{err('i', 'I')} like <b>{err('teh', 'the')}</b> sea{err(' ', ', ')}too</p>")[0]
    assert corrected_text(p) == "I like the sea, too"


def test_corrected_text_ignores_nested_paragraphs():
    p = doc(f"<p>{err('a', 'A')} <p>{err('b', 'B')}</p> c</p>")[0]
    assert corrected_text(p) == "A  c"


def test_assignment_expected_output():
    root = etree.parse(str(SAMPLES / "expected_output.xml")).getroot()
    first = next(root.iter("{*}p"))
    assert corrected_text(first) == "For breakfast, I like eggs, toast, and juice."


def test_identical_outputs_score_perfectly():
    reference = etree.parse(str(SAMPLES / "expected_output.xml")).getroot()
    result = score(reference, reference)
    assert (result.precision, result.recall, result.exact) == (1.0, 1.0, 5)


def test_combined_and_split_fixes_score_the_same():
    # One <error> for "paris france", or three: same corrected text, same score.
    combined = doc(f"<p>{err('paris france', 'Paris, France')}</p>")
    split = doc(f"<p>{err('paris', 'Paris')}{err(' ', ', ')}{err('france', 'France')}</p>")
    result = score(combined, split)
    assert (result.precision, result.recall, result.exact) == (1.0, 1.0, 1)


def test_wrong_and_missed_edits_are_counted():
    reference = doc(f"<p>{err('i', 'I')} saw {err('paris', 'Paris')}.</p>")
    output = doc(f"<p>{err('i', 'I')} {err('saw', 'see')} paris.</p>")
    result = score(reference, output)
    assert (result.hits, result.extra, result.missed, result.exact) == (1, 1, 1, 0)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)


def test_no_edits_at_all_is_perfect():
    result = score(doc("<p>Fine.</p>"), doc("<p>Fine.</p>"))
    assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)


def test_edits_are_token_level():
    assert edits("eggs toast", "eggs, toast") == {(1, 1, ",")}


def test_documents_must_match():
    with pytest.raises(ValueError):
        score(doc("<p>a</p>"), doc("<p>a</p><p>b</p>"))
