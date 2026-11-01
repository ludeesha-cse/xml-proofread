"""End-to-end tests of pipeline.run() with a fake LLM."""

from pathlib import Path

import pytest

from proofread.config import build_settings
from proofread.exceptions import LLMError
from proofread.pipeline import run

SAMPLES = Path(__file__).parent.parent / "samples"

# What a good model answers for samples/input.xml, as (first, last, type, correction).
SAMPLE_ANSWERS = {
    "For breakfast I like eggs toast and juice.": [
        (3, 3, "punctuation", ", "),
        (9, 9, "punctuation", ", "),
        (11, 12, "punctuation", ", and"),
    ],
    "The companys employees work hard but their unhappy.": [
        (2, 2, "punctuation", "company's"),
        (9, 9, "punctuation", ", "),
        (12, 12, "grammar", "they're"),
    ],
    "sarah lives in paris france? She enjoy french food.": [
        (0, 0, "capitalization", "Sarah"),
        (6, 8, "capitalization", "Paris, France"),
        (9, 9, "punctuation", "."),
        (13, 13, "grammar", "enjoys"),
    ],
    "i visited london last summer and i loved the museums the galleries and the food.": [
        (0, 0, "capitalization", "I"),
        (4, 4, "capitalization", "London"),
        (12, 12, "capitalization", "I"),
        (19, 19, "punctuation", ", "),
        (23, 23, "punctuation", ", "),
    ],
}


def settings_for(src: Path, **overrides):
    values = dict(
        input_path=src, lang="en", model="ollama:fake", output_path=None, batch_size=2,
        concurrency=1, max_retries=0, timeout_s=10.0, strict=False,
    )  # fmt: skip
    values.update(overrides)
    return build_settings(**values)


@pytest.fixture
def sample_copy(tmp_path) -> Path:
    src = tmp_path / "input.xml"
    src.write_bytes((SAMPLES / "input.xml").read_bytes())
    return src


def test_pipeline_reproduces_the_expected_output(sample_copy, fake_llm):
    llm = fake_llm(SAMPLE_ANSWERS)
    result = run(settings_for(sample_copy), llm=llm)

    expected = (SAMPLES / "expected_output.xml").read_bytes()
    assert result.output_path.read_bytes() == expected
    assert (result.paragraphs, result.proofread, result.errors_applied) == (5, 5, 15)
    assert (result.errors_rejected, result.failed) == (0, 0)
    assert [len(call) for call in llm.calls] == [2, 2, 1]  # batch_size=2


def test_pipeline_with_no_errors_writes_identical_copy(sample_copy, fake_llm):
    result = run(settings_for(sample_copy), llm=fake_llm({}))
    assert result.output_path.read_bytes() == sample_copy.read_bytes()


def test_pipeline_fails_when_every_request_fails(sample_copy, fake_llm):
    with pytest.raises(LLMError, match="Every LLM request failed"):
        run(settings_for(sample_copy), llm=fake_llm(fail=True))
    assert not (sample_copy.parent / "input.corrected.xml").exists()


def test_pipeline_reports_timings_and_memory(sample_copy, fake_llm):
    result = run(settings_for(sample_copy), llm=fake_llm({}))
    assert list(result.stage_seconds) == ["parse", "proofread", "annotate", "write"]
    assert all(seconds >= 0 for seconds in result.stage_seconds.values())
    assert result.total_seconds >= sum(result.stage_seconds.values())
    assert result.requests == 3  # 5 paragraphs, batch_size=2
    assert result.peak_memory_mb is None or result.peak_memory_mb > 0
