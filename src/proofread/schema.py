"""The shape of the answer we ask the LLM to return.

LangChain turns these Pydantic classes into a JSON schema and asks the model to follow
it (structured output). Gemini and Ollama both enforce the schema while generating, so
the reply is always valid JSON with these exact fields.
The Field descriptions are sent to the model, so they are written as instructions.
"""

from pydantic import BaseModel, Field

from proofread.models import ErrorType


class FoundError(BaseModel):
    """One error the model found."""

    first_token: int = Field(description="Number of the first token of the wrong text.")
    last_token: int = Field(
        description="Number of the last token of the wrong text. Same as first_token "
        "when the error is a single token."
    )
    original: str = Field(
        description="The exact text of the selected tokens, copied character for character."
    )
    type: ErrorType
    correction: str = Field(description="Replacement for exactly the selected tokens.")


class ParagraphErrors(BaseModel):
    """All errors in one paragraph."""

    id: int = Field(description="The paragraph id from the input.")
    errors: list[FoundError] = Field(
        default_factory=list, description="Errors in this paragraph. Empty if none."
    )


class ProofreadResponse(BaseModel):
    """The model's whole answer for one batch of paragraphs."""

    paragraphs: list[ParagraphErrors]


# Extra credit: ask the model for a short reason per error. Off by default (longer answers
# are slower and more expensive); turn on with PROOFREAD_REASONS=true (see README).
class FoundErrorWithReason(FoundError):
    reason: str = Field(description="Why this is an error, in at most 12 words.")


class ParagraphErrorsWithReason(ParagraphErrors):
    errors: list[FoundErrorWithReason] = Field(
        default_factory=list, description="Errors in this paragraph. Empty if none."
    )


class ProofreadResponseWithReason(ProofreadResponse):
    paragraphs: list[ParagraphErrorsWithReason]


def response_schema(include_reasons: bool) -> type[ProofreadResponse]:
    """The schema to ask the model for: with or without a reason per error."""
    return ProofreadResponseWithReason if include_reasons else ProofreadResponse
