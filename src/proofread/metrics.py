"""Measuring a run: time per stage, peak memory and LLM token usage."""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.callbacks import UsageMetadataCallbackHandler


class Stopwatch:
    """Times named stages of a run.

    with stopwatch.stage("parse"):
        ...  # the time spent here is added to stopwatch.seconds["parse"]
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start

    @property
    def total(self) -> float:
        return time.perf_counter() - self.started


def peak_memory_mb() -> float | None:
    """Highest memory this process has used so far (peak RSS), in MB.

    Uses the standard `resource` module, which costs nothing to call. It does not
    exist on Windows, where None is returned.
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024


def token_totals(counter: UsageMetadataCallbackHandler) -> tuple[int, int]:
    """(input tokens, output tokens) summed over every model the counter saw."""
    usage = counter.usage_metadata.values()
    return (
        sum(u.get("input_tokens", 0) for u in usage),
        sum(u.get("output_tokens", 0) for u in usage),
    )
