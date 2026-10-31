"""Small data types shared by several modules."""

from dataclasses import dataclass
from typing import Literal, get_args

# The five error categories the assignment asks for. Anything else is rejected.
# ErrorType is used in the LLM's output schema; ERROR_TYPES is the same list as a tuple.
ErrorType = Literal["grammar", "spelling", "punctuation", "capitalization", "clarity"]
ERROR_TYPES: tuple[str, ...] = get_args(ErrorType)

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
