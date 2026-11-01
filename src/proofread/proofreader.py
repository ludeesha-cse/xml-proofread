"""Send paragraphs to the LLM in batches and turn its answers into ErrorSpans.

The model answers in token numbers. This module checks those numbers against the
tokens we sent, converts them to character positions, and handles failed requests.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from langchain_core.runnables import Runnable
from lxml import etree

from proofread.exceptions import LLMError
from proofread.models import ErrorSpan
from proofread.paragraph import contains_cdata, text_slots
from proofread.prompts import build_messages
from proofread.schema import FoundError, ProofreadResponse
from proofread.tokenizer import Token, token_range_to_chars, tokenize

logger = logging.getLogger(__name__)


@dataclass
class ParagraphJob:
    """One paragraph, prepared for the model."""

    index: int  # position in the document's list of <p> elements
    text: str
    tokens: list[Token]


@dataclass
class ProofreadOutput:
    """Everything the model found, keyed by paragraph index."""

    spans: dict[int, list[ErrorSpan]] = field(default_factory=dict)
    dropped: int = 0  # suggestions we could not map to the text
    failed: list[int] = field(default_factory=list)  # paragraphs whose request failed
    requests: int = 0  # LLM requests made (not counting retries)
    first_error: str | None = None  # why the first failed request failed, for messages
    duplicates: int = 0  # paragraphs not sent because an identical one was


def make_job(index: int, p: etree._Element) -> ParagraphJob | None:
    """Prepare a paragraph for the model, or return None if there is nothing to check."""
    if contains_cdata(p):
        logger.debug("Skipping paragraph %d: contains CDATA", index)
        return None
    pieces = [slot.value for slot in text_slots(p)]
    text = "".join(pieces)
    if not any(ch.isalpha() for ch in text):
        return None  # empty, whitespace or numbers only: nothing to proofread
    return ParagraphJob(index=index, text=text, tokens=tokenize(pieces))


def _find_tokens(tokens: list[Token], original: str) -> tuple[int, int] | None:
    """Find the only token range whose text equals `original`, if there is exactly one."""
    matches = []
    for first in range(len(tokens)):
        joined = ""
        for last in range(first, len(tokens)):
            joined += tokens[last].text
            if joined == original:
                matches.append((first, last))
            if len(joined) >= len(original):
                break
    return matches[0] if len(matches) == 1 else None


# Punctuation that never has a space before it (also true in French, unlike ; : ! ?).
_NO_SPACE_BEFORE = (",", ".")


def _fix_space_before_punctuation(text: str, start: int, correction: str) -> tuple[int, str]:
    """Repair a common model slip that would put a space before a comma.

    For "galleries and" the model should select the space and correct it to ", ".
    Sometimes it selects "and" and corrects it to ", and" instead, which leaves the
    space in front: "galleries , and". If a correction starts with a comma or period,
    we drop any leading whitespace from it and widen the span to swallow the
    whitespace before it, giving "galleries, and".
    """
    stripped = correction.lstrip()
    if not stripped.startswith(_NO_SPACE_BEFORE):
        return start, correction
    while start > 0 and text[start - 1].isspace():
        start -= 1
    return start, stripped


def to_error_span(job: ParagraphJob, found: FoundError) -> ErrorSpan:
    """Convert one model suggestion into an ErrorSpan. Raises ValueError if impossible.

    Models sometimes get a token number off by one. The `original` field lets us notice:
    if the selected tokens don't spell `original`, we look for `original` in the
    paragraph and use it if it appears exactly once.
    """
    first, last = found.first_token, found.last_token
    selected = "".join(t.text for t in job.tokens[first : last + 1]) if first <= last else ""
    if selected != found.original:
        fixed = _find_tokens(job.tokens, found.original)
        if fixed is None:
            raise ValueError(
                f"tokens {first}-{last} are {selected!r}, not {found.original!r}, "
                "and that text could not be located unambiguously"
            )
        logger.debug("Corrected token range %d-%d to %d-%d", first, last, *fixed)
        first, last = fixed

    start, end = token_range_to_chars(job.tokens, first, last)
    start, correction = _fix_space_before_punctuation(job.text, start, found.correction)
    return ErrorSpan(
        start=start,
        end=end,
        type=found.type,
        correction=correction,
        reason=getattr(found, "reason", None),  # only present if the extra credit is on
    )


class Proofreader:
    """Runs batches of ParagraphJobs through a structured-output LLM, in parallel."""

    def __init__(
        self,
        llm: Runnable,
        *,
        language: str,
        lang: str,
        batch_size: int,
        concurrency: int,
        strict: bool,
        callbacks: list | None = None,
    ) -> None:
        self._llm = llm  # returns a ProofreadResponse (see llm.build_structured_llm)
        self._language = language
        self._lang = lang
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._strict = strict
        self._callbacks = callbacks or []  # e.g. a token-usage counter

    async def proofread(self, jobs: list[ParagraphJob]) -> ProofreadOutput:
        output = ProofreadOutput()

        # Identical paragraphs (same tokens) get identical answers: send each once.
        groups: dict[tuple[str, ...], list[ParagraphJob]] = {}
        for job in jobs:
            groups.setdefault(tuple(t.text for t in job.tokens), []).append(job)
        unique = [same[0] for same in groups.values()]
        output.duplicates = len(jobs) - len(unique)

        # Run the batches in parallel, but never more than `concurrency` at once.
        batches = [
            unique[i : i + self._batch_size] for i in range(0, len(unique), self._batch_size)
        ]
        semaphore = asyncio.Semaphore(self._concurrency)

        async def run_limited(batch: list[ParagraphJob]) -> None:
            async with semaphore:
                await self._run_batch(batch, output)

        await asyncio.gather(*(run_limited(batch) for batch in batches))

        # Copy each answer to the duplicates of that paragraph.
        for first, *copies in groups.values():
            for copy in copies:
                if first.index in output.spans:
                    output.spans[copy.index] = list(output.spans[first.index])
                if first.index in output.failed:
                    output.failed.append(copy.index)
        return output

    async def _run_batch(self, batch: list[ParagraphJob], output: ProofreadOutput) -> None:
        messages = build_messages(
            self._language, self._lang, [(job.text, job.tokens) for job in batch]
        )
        output.requests += 1
        try:
            response: ProofreadResponse = await self._llm.ainvoke(
                messages, config={"callbacks": self._callbacks}
            )
        except Exception as exc:
            indexes = [job.index for job in batch]
            if self._strict:
                raise LLMError(f"LLM request failed for paragraphs {indexes}: {exc}") from exc
            logger.warning("LLM request failed; leaving paragraphs %s unchanged: %s", indexes, exc)
            output.failed += indexes
            output.first_error = output.first_error or f"{type(exc).__name__}: {exc}"
            return

        # The prompt asks for every paragraph id. A missing one could mean "no errors"
        # or "forgot it", and we can't tell which, so treat it as not proofread.
        returned = {answer.id for answer in response.paragraphs}
        missing = [job.index for i, job in enumerate(batch) if i not in returned]
        if missing:
            if self._strict:
                raise LLMError(f"The model skipped paragraphs {missing}")
            logger.warning("The model skipped paragraphs %s; leaving them unchanged", missing)
            output.failed += missing

        for answer in response.paragraphs:
            if not 0 <= answer.id < len(batch):
                logger.debug("Ignoring answer for unknown paragraph id %d", answer.id)
                continue
            job = batch[answer.id]
            for found in answer.errors:
                try:
                    span = to_error_span(job, found)
                except ValueError as exc:
                    output.dropped += 1
                    logger.debug("Dropped suggestion in paragraph %d: %s", job.index, exc)
                    continue
                output.spans.setdefault(job.index, []).append(span)
