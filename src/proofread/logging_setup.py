"""Logging configuration for the whole program."""

import logging
import sys

# Example line: 21:10:55 INFO    proofread: Model: ollama:qwen3:8b
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Libraries that log every HTTP request at DEBUG level. We keep them quiet
# so --verbose shows *our* debug messages, not thousands of network lines.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "google", "google_genai", "langchain")


def setup_logging(verbose: bool = False) -> None:
    """Send logs to stderr. Level is DEBUG with --verbose, otherwise INFO."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stderr,  # stdout stays free for the final summary
        force=True,  # replace any handlers a library already added
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
