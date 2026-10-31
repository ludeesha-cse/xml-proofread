"""Small data types shared by several modules."""

from dataclasses import dataclass

# The five error categories
ERROR_TYPES = ("grammar", "spelling", "punctuation", "capitalization", "clarity")


@dataclass(frozen=True)
class ErrorSpan:
    """One mistake in a paragraph's text.

    start/end are character positions in the paragraph's text, like a Python slice:
    text[start:end] is the original text that will be wrapped in <error>.
    """

    start: int
    end: int
    type: str  # one of ERROR_TYPES
    correction: str  # what the wrapped text should be replaced with
    reason: str | None = None  # extra credit: short explanation (see injector.py)
