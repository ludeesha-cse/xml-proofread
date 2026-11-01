"""Wrap mistakes in <error> tags without changing the paragraph's text.

The LLM only *suggests* ErrorSpans. This module decides which ones are safe, edits the
lxml tree, then checks the text is still identical. The model never writes XML.
"""

import logging
from dataclasses import dataclass, field
from itertools import pairwise

from lxml import etree

from proofread.models import ERROR_TYPES, ErrorSpan
from proofread.paragraph import TextSlot, contains_cdata, paragraph_text, text_slots

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResult:
    """What happened to one paragraph's suggested errors."""

    applied: list[ErrorSpan] = field(default_factory=list)
    rejected: list[tuple[ErrorSpan, str]] = field(default_factory=list)  # (span, reason)


def _make_error(parent: etree._Element, span: ErrorSpan, original: str) -> etree._Element:
    """Create <error type=".." correction="..">original</error> (not yet in the tree)."""
    # Use the parent's default namespace, if it has one, so the tag is written as a
    # plain <error> and still belongs to the same namespace as its neighbours.
    namespace = parent.nsmap.get(None)
    tag = f"{{{namespace}}}error" if namespace else "error"

    error = parent.makeelement(tag, {"type": span.type, "correction": span.correction})
    # Extra credit: a short reason for each error. It is collected from the model but not
    # written by default. To enable it, uncomment the two lines below (see README).
    # if span.reason:
    #     error.set("reason", span.reason)
    error.text = original
    return error


def _wrap_slot(slot: TextSlot, spans: list[ErrorSpan]) -> list[etree._Element]:
    """Split one text slot into: text before, <error>, text between, <error>, ..., rest.

    Example: slot "eggs toast and juice" with one span on " and":
        before = "eggs toast", <error> " and" </error>, its tail = " juice"
    """
    value = slot.value
    # Convert paragraph positions into positions inside this slot.
    local = [(span, span.start - slot.start, span.end - slot.start) for span in spans]

    errors = []
    for i, (span, start, end) in enumerate(local):
        next_start = local[i + 1][1] if i + 1 < len(local) else len(value)
        parent = slot.owner if slot.attr == "text" else slot.owner.getparent()
        error = _make_error(parent, span, value[start:end])
        error.tail = value[end:next_start] or None  # text up to the next error
        errors.append(error)

    # Keep only the text before the first error in the original slot...
    setattr(slot.owner, slot.attr, value[: local[0][1]] or None)
    # ...and insert the errors right after it.
    if slot.attr == "text":
        slot.owner.insert(0, errors[0])  # first child of the element
    else:
        slot.owner.addnext(errors[0])  # next sibling, after the owner's tail
    for previous, error in pairwise(errors):
        previous.addnext(error)
    return errors


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


def _undo(inserted: list[etree._Element], saved: list[tuple[TextSlot, str]]) -> None:
    """Remove the inserted <error> elements and restore the saved original text.

    We restore from saved copies instead of rebuilding from the <error> elements, so
    even a buggy <error> can't leak wrong text back into the document.
    """
    for error in inserted:
        error.getparent().remove(error)  # in lxml, the error's tail is removed with it
    for slot, value in saved:
        setattr(slot.owner, slot.attr, value or None)


def annotate_paragraph(p: etree._Element, spans: list[ErrorSpan]) -> AnnotationResult:
    """Wrap each valid span of `p` in an <error> tag. The text never changes.

    1. Reject unsafe spans (validate_spans).
    2. Edit the tree.
    3. Check the text is identical. If not (a bug), undo every change.
    """
    if not spans:
        return AnnotationResult()
    if contains_cdata(p):
        return AnnotationResult(rejected=[(s, "paragraph contains CDATA") for s in spans])

    slots = text_slots(p)
    original_text = paragraph_text(p)
    accepted, rejected = validate_spans(spans, original_text, slots)
    for span, reason in rejected:
        logger.debug("Rejected %s: %s", span, reason)

    inserted: list[etree._Element] = []
    saved: list[tuple[TextSlot, str]] = []  # original value of every slot we change
    for slot in slots:
        in_slot = [s for s in accepted if slot.start <= s.start and s.end <= slot.end]
        if in_slot:
            saved.append((slot, slot.value))
            inserted += _wrap_slot(slot, in_slot)

    # The length invariant: the text without tags must be exactly the same.
    if paragraph_text(p) != original_text:
        _undo(inserted, saved)
        logger.error("Text changed while annotating a paragraph; changes undone.")
        rejected += [(span, "failed the text check") for span in accepted]
        return AnnotationResult(rejected=rejected)

    return AnnotationResult(applied=accepted, rejected=rejected)
