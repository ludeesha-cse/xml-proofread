"""The end-to-end flow for one file: load -> find <p> -> proofread -> inject -> save.

The CLI calls run(). Keeping the flow here, not in cli.py, means tests can run the
whole pipeline directly, with a fake LLM instead of a real one.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.runnables import Runnable

from proofread.config import Settings
from proofread.exceptions import LLMError
from proofread.injector import annotate_paragraph
from proofread.llm import build_structured_llm, create_chat_model
from proofread.metrics import Stopwatch, peak_memory_mb, token_totals
from proofread.paragraph import has_error_tags
from proofread.proofreader import ParagraphJob, Proofreader, ProofreadOutput, make_job
from proofread.xml_io import find_paragraphs, load_xml, save_xml

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """What happened during one run: counts, timings and memory."""

    output_path: Path
    paragraphs: int  # <p> elements in the document
    proofread: int  # paragraphs that needed the model
    duplicates: int  # of those, answered from an identical paragraph
    errors_applied: int  # <error> tags written
    errors_rejected: int  # suggestions we could not use safely
    failed: int  # paragraphs left unchanged because the request failed
    requests: int  # LLM requests made
    input_tokens: int
    output_tokens: int
    stage_seconds: dict[str, float]  # time per stage: parse, proofread, annotate, write
    total_seconds: float
    peak_memory_mb: float | None  # None on Windows
    failed: int  # paragraphs left unchanged because the request failed


async def close_model(model: object) -> None:
    """Close the model's network connections while the event loop is still running.

    Gemini's client closes its connections on the event loop it used. If that loop has
    already finished (asyncio.run returned), Python later logs "Event loop is closed".
    Models without aclose() (e.g. Ollama) need nothing.
    """
    aclose = getattr(model, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as exc:  # cleanup must never hide the real result
        logger.debug("Closing the model failed: %s", exc)


def _proofread(
    settings: Settings,
    llm: Runnable | None,
    jobs: list[ParagraphJob],
    token_counter: UsageMetadataCallbackHandler,
) -> ProofreadOutput:
    """Create the model (unless a test passed one in) and proofread every job."""
    chat_model = None
    if llm is None:
        chat_model = create_chat_model(settings)
        llm = build_structured_llm(chat_model, settings)
    proofreader = Proofreader(
        llm,
        language=settings.language_name,
        lang=settings.lang,
        batch_size=settings.batch_size,
        concurrency=settings.concurrency,
        strict=settings.strict,
        callbacks=[token_counter],
    )

    async def proofread_then_close() -> ProofreadOutput:
        try:
            return await proofreader.proofread(jobs)
        finally:
            await close_model(chat_model)

    return asyncio.run(proofread_then_close())


def run(settings: Settings, llm: Runnable | None = None) -> RunResult:
    """Proofread one file. Pass `llm` to use a fake model in tests."""
    stopwatch = Stopwatch()

    with stopwatch.stage("parse"):
        doc = load_xml(settings.input_path)
        paragraphs = find_paragraphs(doc.root)
        jobs = [job for i, p in enumerate(paragraphs) if (job := make_job(i, p))]
    logger.info("Found %d <p> element(s); %d need proofreading", len(paragraphs), len(jobs))
    if already := sum(has_error_tags(p) for p in paragraphs):
        logger.warning(
            "%d paragraph(s) already contain <error> tags (is this an output file?); "
            "they were left unchanged",
            already,
        )

    token_counter = UsageMetadataCallbackHandler()  # LangChain adds up token usage here
    found = ProofreadOutput()
    with stopwatch.stage("proofread"):
        if jobs:  # nothing to check means no model is needed (no API key, no Ollama)
            found = _proofread(settings, llm, jobs, token_counter)

    if jobs and len(found.failed) == len(jobs):
        # Nothing worked at all: almost certainly a setup problem (Ollama not running,
        # wrong model name, bad API key). Fail loudly instead of writing an unchanged copy.
        raise LLMError(
            f"Every LLM request failed ({found.first_error}). "
            "Check the model name, the API key, or that Ollama is running."
        )

    applied = 0
    rejected = found.dropped
    with stopwatch.stage("annotate"):
        for index, spans in found.spans.items():
            result = annotate_paragraph(paragraphs[index], spans)
            applied += len(result.applied)
            rejected += len(result.rejected)

    with stopwatch.stage("write"):
        save_xml(doc, settings.output_path)
    logger.info("Wrote %s", settings.output_path)

    input_tokens, output_tokens = token_totals(token_counter)
    return RunResult(
        output_path=settings.output_path,
        paragraphs=len(paragraphs),
        proofread=len(jobs),
        duplicates=found.duplicates,
        errors_applied=applied,
        errors_rejected=rejected,
        failed=len(found.failed),
        requests=found.requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stage_seconds=stopwatch.seconds,
        total_seconds=stopwatch.total,
        peak_memory_mb=peak_memory_mb(),
    )
