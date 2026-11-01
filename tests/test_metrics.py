"""Tests for metrics.py."""

import time

from langchain_core.callbacks import UsageMetadataCallbackHandler

from proofread.metrics import Stopwatch, peak_memory_mb, token_totals


def test_stopwatch_adds_up_repeated_stages():
    stopwatch = Stopwatch()
    for _ in range(2):
        with stopwatch.stage("work"):
            time.sleep(0.01)
    assert 0.02 <= stopwatch.seconds["work"] < 0.5
    assert stopwatch.total >= stopwatch.seconds["work"]


def test_stopwatch_records_time_even_if_the_stage_fails():
    stopwatch = Stopwatch()
    try:
        with stopwatch.stage("broken"):
            raise ValueError
    except ValueError:
        pass
    assert "broken" in stopwatch.seconds


def test_peak_memory_is_plausible():
    mb = peak_memory_mb()
    assert mb is None or 1 < mb < 10_000


def test_token_totals_sum_every_model():
    counter = UsageMetadataCallbackHandler()
    counter.usage_metadata = {
        "model-a": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "model-b": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    }
    assert token_totals(counter) == (105, 21)
