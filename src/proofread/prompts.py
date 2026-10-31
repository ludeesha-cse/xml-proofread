"""The instructions and input we send to the LLM.

The system message explains the task once. The human message holds the paragraphs,
each shown twice: as plain text (so the model can read it naturally) and as numbered
tokens (so it can point at errors precisely).
"""

import json
from string import Template

from langchain_core.messages import HumanMessage, SystemMessage

from proofread.tokenizer import Token

# $language and $lang are filled in per run. string.Template is used instead of an
# f-string so the JSON example below can use { } without escaping.
_SYSTEM_TEMPLATE = Template("""\
You are a careful proofreader for $language text (language tag: $lang).

You receive paragraphs. Each paragraph is split into numbered tokens: words, single
punctuation marks, and runs of whitespace. Find real errors and report each one by the
numbers of its first and last token.

Error types:
- grammar: agreement, verb tense or form, wrong word (their/they're), missing or wrong articles.
- spelling: misspelled words.
- punctuation: missing, extra or wrong punctuation marks, including apostrophes.
- capitalization: missing or wrong capital letters (sentence starts, proper nouns, "I").
- clarity: only text that is genuinely confusing, such as a dangling modifier or an
  accidentally repeated word. Never report style preferences.

Rules:
1. "correction" replaces exactly the selected tokens, including any whitespace tokens
   you select. Everything else in the paragraph stays as it is.
2. Select as few tokens as possible.
3. For missing punctuation, select the whitespace token where it belongs and include
   the space in the correction, e.g. token " " becomes ", ".
4. When neighbouring words need one combined fix, select them together as one error.
5. Errors must not overlap.
6. "original" must be the exact text of the selected tokens.
7. Follow the spelling, punctuation and capitalization rules of $language. Do not change
   valid regional spellings or text that is already correct.
8. A paragraph without errors gets an empty list. Never invent errors.
9. Return an entry for every paragraph id you received.

Example input:
<paragraph id="0">
text: "For breakfast I like eggs toast and juice."
tokens: 0:"For" 1:" " 2:"breakfast" 3:" " 4:"I" 5:" " 6:"like" 7:" " 8:"eggs" 9:" " 10:"toast" 11:" " 12:"and" 13:" " 14:"juice" 15:"."
</paragraph>

Example answer:
{"paragraphs": [{"id": 0, "errors": [
  {"first_token": 9, "last_token": 9, "original": " ", "type": "punctuation", "correction": ", "},
  {"first_token": 11, "last_token": 12, "original": " and", "type": "punctuation", "correction": ", and"}
]}]}""")  # noqa: E501  (the example lines must stay exactly as the model sees them)


def system_prompt(language: str, lang: str) -> str:
    return _SYSTEM_TEMPLATE.substitute(language=language, lang=lang)


def format_paragraph(paragraph_id: int, text: str, tokens: list[Token]) -> str:
    """One paragraph in the format shown in the system prompt's example.

    json.dumps quotes every value and escapes quotes and newlines inside it, so a
    token like " or a line break can never confuse the format.
    """
    numbered = " ".join(
        f"{i}:{json.dumps(token.text, ensure_ascii=False)}" for i, token in enumerate(tokens)
    )
    return (
        f'<paragraph id="{paragraph_id}">\n'
        f"text: {json.dumps(text, ensure_ascii=False)}\n"
        f"tokens: {numbered}\n"
        f"</paragraph>"
    )


def build_messages(
    language: str, lang: str, paragraphs: list[tuple[str, list[Token]]]
) -> list[SystemMessage | HumanMessage]:
    """The full chat for one batch. Paragraph ids are their positions in the batch."""
    body = "\n\n".join(
        format_paragraph(i, text, tokens) for i, (text, tokens) in enumerate(paragraphs)
    )
    return [
        SystemMessage(system_prompt(language, lang)),
        HumanMessage(f"Proofread these paragraphs.\n\n{body}"),
    ]
