"""The end-to-end flow for one file: load -> find <p> -> proofread -> inject -> save.

The CLI calls run(). Keeping the flow here, not in cli.py, means tests can run the
whole pipeline directly, with a fake LLM instead of a real one.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.runnables import Runnable

from proofread.config import Settings
from proofread.exceptions import LLMError
from proofread.injector import annotate_paragraph
from proofread.llm import build_structured_llm, create_chat_model
from proofread.proofreader import Proofreader, make_job
from proofread.xml_io import find_paragraphs, load_xml, save_xml

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """What happened during one run. #todo: adds timings and memory."""

    output_path: Path
    paragraphs: int  # <p> elements in the document
    proofread: int  # paragraphs sent to the model
    errors_applied: int  # <error> tags written
    errors_rejected: int  # suggestions we could not use safely
    failed: int  # paragraphs left unchanged because the request failed


def run(settings: Settings, llm: Runnable | None = None) -> RunResult:
    """Proofread one file. Pass `llm` to use a fake model in tests."""
    doc = load_xml(settings.input_path)
    paragraphs = find_paragraphs(doc.root)
    jobs = [job for i, p in enumerate(paragraphs) if (job := make_job(i, p))]
    logger.info("Found %d <p> element(s); %d need proofreading", len(paragraphs), len(jobs))

    if llm is None:
        llm = build_structured_llm(create_chat_model(settings), settings)
    proofreader = Proofreader(
        llm,
        language=settings.language_name,
        lang=settings.lang,
        batch_size=settings.batch_size,
        concurrency=settings.concurrency,
        strict=settings.strict,
    )
    found = asyncio.run(proofreader.proofread(jobs))

    if jobs and len(found.failed) == len(jobs):
        # Nothing worked at all: almost certainly a setup problem (Ollama not running,
        # wrong model name, bad API key). Fail loudly instead of writing an unchanged copy.
        raise LLMError("Every LLM request failed. Check the model name, API key or Ollama.")

    applied = 0
    rejected = found.dropped
    for index, spans in found.spans.items():
        result = annotate_paragraph(paragraphs[index], spans)
        applied += len(result.applied)
        rejected += len(result.rejected)

    save_xml(doc, settings.output_path)
    logger.info(
        "Wrote %s: %d error(s) marked, %d suggestion(s) rejected, %d paragraph(s) failed",
        settings.output_path, applied, rejected, len(found.failed),
    )  # fmt: skip
    return RunResult(
        output_path=settings.output_path,
        paragraphs=len(paragraphs),
        proofread=len(jobs),
        errors_applied=applied,
        errors_rejected=rejected,
        failed=len(found.failed),
    )
