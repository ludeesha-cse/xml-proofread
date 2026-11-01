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
from proofread.proofreader import Proofreader, make_job
from proofread.xml_io import find_paragraphs, load_xml, save_xml
from proofread.metrics import Stopwatch, peak_memory_mb, token_totals

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


def run(settings: Settings, llm: Runnable | None = None) -> RunResult:
    """Proofread one file. Pass `llm` to use a fake model in tests."""
    stopwatch = Stopwatch()

    with stopwatch.stage("parse"):
        doc = load_xml(settings.input_path)
        paragraphs = find_paragraphs(doc.root)
        jobs = [job for i, p in enumerate(paragraphs) if (job := make_job(i, p))]
    logger.info("Found %d <p> element(s); %d need proofreading", len(paragraphs), len(jobs))

    if llm is None:
        llm = build_structured_llm(create_chat_model(settings), settings)
    token_counter = UsageMetadataCallbackHandler()  # LangChain adds up token usage here
    proofreader = Proofreader(
        llm,
        language=settings.language_name,
        lang=settings.lang,
        batch_size=settings.batch_size,
        concurrency=settings.concurrency,
        strict=settings.strict,
        callbacks=[token_counter],
    )
    with stopwatch.stage("proofread"):
        found = asyncio.run(proofreader.proofread(jobs))
    if jobs and len(found.failed) == len(jobs):
        # Nothing worked at all: almost certainly a setup problem (Ollama not running,
        # wrong model name, bad API key). Fail loudly instead of writing an unchanged copy.
        raise LLMError("Every LLM request failed. Check the model name, API key or Ollama.")

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
