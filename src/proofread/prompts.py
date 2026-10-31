"""The instructions and input we send to the LLM.

The system message explains the task once. The human message holds the paragraphs,
each shown twice: as plain text (so the model can read it naturally) and as numbered
tokens (so it can point at errors precisely).
"""

import json
from string import Template

from langchain_core.messages import HumanMessage, SystemMessage

from proofread.tokenizer import Token, tokenize

# Worked examples shown to the model: (paragraph, [(first, last, type, correction)]).
# They were written for this prompt and are NOT taken from samples/input.xml, so the
# sample stays a fair test. Token numbers come from our tokenizer (see _render_examples).
EXAMPLES: list[tuple[str, list[tuple[int, int, str, str]]]] = [
    (
        "my brother live in tokyo japan.",
        [
            (0, 0, "capitalization", "My"),
            (4, 4, "grammar", "lives"),
            # One combined fix: two capitals AND the missing comma between them.
            (8, 10, "capitalization", "Tokyo, Japan"),
        ],
    ),
    ("The report is ready, and the team will review it tomorrow.", []),
    (
        "Its a realy reliable tool so we use it every day.",
        [
            (0, 0, "punctuation", "It's"),
            (4, 4, "spelling", "really"),
            (9, 9, "punctuation", ", "),  # missing comma: select the space
        ],
    ),
]

# $language, $lang and $examples are filled in per run. string.Template is used
# instead of an f-string so the JSON in the examples can use { } freely.
_SYSTEM_TEMPLATE = Template("""\
You are a careful proofreader for $language text (language tag: $lang).

You receive paragraphs. Each paragraph is split into numbered tokens: words, single
punctuation marks, and runs of whitespace. Find real errors and report each one by the
numbers of its first and last token.

Error types:
- grammar: agreement, verb tense or form, wrong word (their/they're), missing or wrong articles.
- spelling: misspelled words.
- punctuation: missing, extra or wrong punctuation, including apostrophes and a wrong
  mark at the end of a sentence.
- capitalization: missing or wrong capital letters (sentence starts, proper nouns, "I").
- clarity: only text that is genuinely confusing, such as a dangling modifier or an
  accidentally repeated word. Never report style preferences.

Rules:
1. "correction" replaces exactly the selected tokens, including any whitespace tokens
   you select. Everything else in the paragraph stays as it is.
2. Select as few tokens as possible, but never split one fix into pieces.
3. For missing punctuation, select the whitespace token where it belongs and include
   the space in the correction: the token " " becomes ", ".
4. When one fix changes several neighbouring tokens, for example two words that both
   need capitals and a missing comma between them, select everything from the first to
   the last of those tokens, including the whitespace between them, and give the whole
   corrected phrase as one correction: "tokyo japan" becomes "Tokyo, Japan".
5. Errors must not overlap.
6. "original" must be the exact text of the selected tokens.
7. Follow the spelling, punctuation and capitalization rules of $language. Do not change
   valid regional spellings.
8. Report only clear errors. If you are not sure something is wrong, leave it. A
   paragraph without errors gets an empty list.
9. Before answering, read each paragraph with all your corrections applied. It must be
   correct, with no missing, doubled or misplaced spaces or punctuation.
10. Return an entry for every paragraph id you received.

$examples""")


def format_paragraph(paragraph_id: int, text: str, tokens: list[Token]) -> str:
    """One paragraph in the format shown in the system prompt's examples.

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


def _render_examples() -> str:
    """Build the example input and answer from EXAMPLES using our real tokenizer.

    Generating them (instead of typing them) guarantees that the token numbers and
    "original" values in the examples always match what the model will see.
    """
    inputs, answers = [], []
    for i, (text, errors) in enumerate(EXAMPLES):
        tokens = tokenize([text])
        inputs.append(format_paragraph(i, text, tokens))
        found = [
            {
                "first_token": first,
                "last_token": last,
                "original": "".join(t.text for t in tokens[first : last + 1]),
                "type": error_type,
                "correction": correction,
            }
            for first, last, error_type, correction in errors
        ]
        answers.append(json.dumps({"id": i, "errors": found}, ensure_ascii=False))
    answer = '{"paragraphs": [\n' + ",\n".join(answers) + "\n]}"
    return "Example input:\n" + "\n\n".join(inputs) + "\n\nExample answer:\n" + answer


def system_prompt(language: str, lang: str) -> str:
    return _SYSTEM_TEMPLATE.substitute(language=language, lang=lang, examples=_render_examples())


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
