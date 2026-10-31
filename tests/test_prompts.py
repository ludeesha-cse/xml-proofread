"""Tests for prompts.py."""

from pathlib import Path

from proofread.prompts import EXAMPLES, build_messages, format_paragraph, system_prompt
from proofread.tokenizer import tokenize


def test_system_prompt_names_the_language():
    prompt = system_prompt("French", "fr")
    assert "proofreader for French text (language tag: fr)" in prompt
    assert "rules of French" in prompt
    assert "$" not in prompt  # every placeholder was filled


def apply_corrections(text: str, errors: list[tuple[int, int, str, str]]) -> str:
    """Apply token-numbered corrections to text, last first so positions stay valid."""
    tokens = tokenize([text])
    for first, last, _, correction in sorted(errors, reverse=True):
        text = text[: tokens[first].start] + correction + text[tokens[last].end :]
    return text


def test_prompt_examples_produce_correct_sentences():
    # If someone edits an example and gets a token number wrong, this fails.
    expected = [
        "My brother lives in Tokyo, Japan.",
        "The report is ready, and the team will review it tomorrow.",
        "It's a really reliable tool, so we use it every day.",
    ]
    assert [apply_corrections(text, errors) for text, errors in EXAMPLES] == expected


def test_prompt_examples_are_not_taken_from_the_sample():
    # The sample file is our test set; showing it to the model would inflate results.
    sample = (Path(__file__).parent.parent / "samples" / "input.xml").read_text()
    for text, _ in EXAMPLES:
        assert text not in sample


def test_system_prompt_contains_the_rendered_examples():
    prompt = system_prompt("English", "en")
    for i, (text, _) in enumerate(EXAMPLES):
        assert format_paragraph(i, text, tokenize([text])) in prompt
    assert '"original": "tokyo japan"' in prompt
    assert '"correction": "Tokyo, Japan"' in prompt


def test_format_paragraph():
    text = "Hi, you."
    assert format_paragraph(3, text, tokenize([text])) == (
        '<paragraph id="3">\n'
        'text: "Hi, you."\n'
        'tokens: 0:"Hi" 1:"," 2:" " 3:"you" 4:"."\n'
        "</paragraph>"
    )


def test_quotes_and_newlines_are_escaped():
    text = 'He said "no"\nthen left.'
    out = format_paragraph(0, text, tokenize([text]))
    assert 'text: "He said \\"no\\"\\nthen left."' in out
    assert '4:"\\""' in out  # the quote character token
    assert '"\\n"' in out  # the newline token


def test_non_ascii_text_is_kept_readable():
    out = format_paragraph(0, "élève", tokenize(["élève"]))
    assert '0:"élève"' in out


def test_build_messages_numbers_paragraphs_in_order():
    paragraphs = [(t, tokenize([t])) for t in ["One.", "Two."]]
    system, human = build_messages("English", "en", paragraphs)
    assert system.type == "system" and human.type == "human"
    assert '<paragraph id="0">\ntext: "One."' in human.content
    assert '<paragraph id="1">\ntext: "Two."' in human.content
