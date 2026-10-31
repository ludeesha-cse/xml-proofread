"""Tests for prompts.py."""

from proofread.prompts import build_messages, format_paragraph, system_prompt
from proofread.tokenizer import tokenize


def test_system_prompt_names_the_language():
    prompt = system_prompt("French", "fr")
    assert "proofreader for French text (language tag: fr)" in prompt
    assert "rules of French" in prompt
    assert "$" not in prompt  # every placeholder was filled


def test_system_prompt_example_matches_our_tokenizer():
    # The example in the prompt must use the same numbering the model will see.
    text = "For breakfast I like eggs toast and juice."
    example = format_paragraph(0, text, tokenize([text]))
    assert example in system_prompt("English", "en")


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
