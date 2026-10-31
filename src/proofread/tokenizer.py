"""Split a paragraph's text into numbered tokens
"""

import re
from dataclasses import dataclass

# A token is one of:
#   a word, with inner apostrophes kept:  company's  they're  l'école
#   a run of whitespace:                  " "  "\n    "
#   a single other character:             ,  .  ?  -  «
# Every character matches one of these, so the tokens always rebuild the text exactly.
_TOKEN_REGAX = re.compile(r"\w+(?:['’]\w+)*|\s+|[^\w\s]")


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # character position in the paragraph text
    end: int


def tokenize(pieces: list[str]) -> list[Token]:
    """Tokenize the paragraph text, given as its slot values in order.

    Each piece is tokenized on its own, so no token ever crosses a slot boundary
    (e.g. the edge of a <b> tag). Positions are counted across all pieces.
    """
    tokens: list[Token] = []
    offset = 0
    for piece in pieces:
        for match in _TOKEN_REGAX.finditer(piece):
            tokens.append(Token(match.group(), offset + match.start(), offset + match.end()))
        offset += len(piece)
    return tokens


def token_range_to_chars(tokens: list[Token], first: int, last: int) -> tuple[int, int]:
    """Convert an inclusive token range into (start, end) character positions.

    Raises ValueError if the range is impossible, e.g. the model invented token 99.
    """
    if not 0 <= first <= last < len(tokens):
        raise ValueError(f"token range {first}-{last} is invalid for {len(tokens)} tokens")
    return tokens[first].start, tokens[last].end
