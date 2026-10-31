"""Wrap mistakes in <error> tags without changing the paragraph's text.

The LLM only *suggests* ErrorSpans. This module decides which ones are safe, edits the
lxml tree, then checks the text is still identical. The model never writes XML.
"""

import logging
from dataclasses import dataclass, field

from proofread.models import ERROR_TYPES, ErrorSpan
from proofread.paragraph import TextSlot

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResult:
    """What happened to one paragraph's suggested errors."""

    applied: list[ErrorSpan] = field(default_factory=list)
    rejected: list[tuple[ErrorSpan, str]] = field(default_factory=list)  # (span, reason)


def validate_spans(
    spans: list[ErrorSpan], text: str, slots: list[TextSlot]
) -> tuple[list[ErrorSpan], list[tuple[ErrorSpan, str]]]:
    """Split suggested spans into (accepted, rejected-with-reason).

    Accepted spans are sorted by position, inside the text, inside one slot, and
    never overlap. These rules are what make the injection safe.
    """
    accepted: list[ErrorSpan] = []
    rejected: list[tuple[ErrorSpan, str]] = []

    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        reason = None
        if span.type not in ERROR_TYPES:
            reason = f"unknown type {span.type!r}"
        elif not 0 <= span.start < span.end <= len(text):
            reason = f"invalid position {span.start}-{span.end}"
        elif span.correction == text[span.start : span.end]:
            reason = "correction is the same as the original"
        elif not any(s.start <= span.start and span.end <= s.end for s in slots):
            reason = "crosses a tag boundary"
        elif accepted and span.start < accepted[-1].end:
            reason = "overlaps another error"

        if reason:
            rejected.append((span, reason))
        else:
            accepted.append(span)

    return accepted, rejected
