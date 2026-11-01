"""Tests that call a real model. Skipped by default (see addopts in pyproject.toml).

Run them with:  uv run pytest -m live -v
The model comes from PROOFREAD_MODEL in your .env (e.g. ollama:qwen3:8b).
"""

from pathlib import Path

import pytest
from dotenv import load_dotenv

from proofread.config import build_settings, require_api_key
from proofread.exceptions import ConfigError
from proofread.paragraph import paragraph_text
from proofread.pipeline import run
from proofread.xml_io import find_paragraphs, load_xml

pytestmark = pytest.mark.live  # applies the "live" mark to every test in this file

SAMPLES = Path(__file__).parent.parent / "samples"


def live_settings(src: Path, lang: str = "en"):
    load_dotenv(override=False)
    settings = build_settings(
        input_path=src, lang=lang, model=None, output_path=None, batch_size=10,
        concurrency=1, max_retries=1, timeout_s=180.0, strict=False,
    )  # fmt: skip
    try:
        require_api_key(settings.provider)
    except ConfigError as exc:
        pytest.skip(str(exc))
    return settings


def assert_text_unchanged(src: Path, out: Path) -> None:
    """The length invariant, checked on the real output file."""
    before = find_paragraphs(load_xml(src).root)
    after = find_paragraphs(load_xml(out).root)
    assert [paragraph_text(p) for p in before] == [paragraph_text(p) for p in after]


def test_real_model_on_the_assignment_sample(tmp_path):
    src = tmp_path / "input.xml"
    src.write_bytes((SAMPLES / "input.xml").read_bytes())

    result = run(live_settings(src))

    assert_text_unchanged(src, result.output_path)
    assert result.failed == 0
    assert result.errors_applied >= 8  # the sample has 15; allow a smaller model to miss some
    corrected = find_paragraphs(load_xml(result.output_path).root)
    clean = corrected[2]  # "They always eat dinner together at home."
    assert len(clean) == 0, "a correct paragraph should get no <error> tags"


def test_real_model_in_french(tmp_path):
    src = tmp_path / "fr.xml"
    src.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<doc><p>hier, je suis allé a paris avec mes amis.</p>"
        "<p>Nous avons visité le musée du Louvre.</p></doc>",
        encoding="utf-8",
    )

    result = run(live_settings(src, lang="fr"))

    assert_text_unchanged(src, result.output_path)
    assert result.errors_applied >= 2  # "hier" -> "Hier", "a" -> "à", "paris" -> "Paris"
