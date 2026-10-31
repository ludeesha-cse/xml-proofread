"""The end-to-end flow for one file: load -> find <p> -> (proofread) -> save.

The CLI calls run(). Keeping the flow here, not in cli.py, means tests can run the
whole pipeline directly without going through the command line.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from proofread.config import Settings
from proofread.xml_io import find_paragraphs, load_xml, save_xml

logger = logging.getLogger(__name__)

@dataclass
class RunResult:
    """What happened during one run. Grows in later phases (errors found, timings)."""

    output_path: Path
    paragraphs: int


def run(settings: Settings) -> RunResult:
    doc = load_xml(settings.input_path)

    paragraphs = find_paragraphs(doc.root)
    logger.info("Found %d <p> element(s)", len(paragraphs))

    # Phases 2-3 plug in here: proofread each paragraph and inject <error> tags.

    save_xml(doc, settings.output_path)
    logger.info("Wrote %s", settings.output_path)
    return RunResult(output_path=settings.output_path, paragraphs=len(paragraphs))
