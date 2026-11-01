"""Tests for config.py: every rule in build_settings() and its helpers."""

from pathlib import Path

import pytest

from proofread.config import (
    DEFAULT_MODEL,
    build_settings,
    default_output_path,
    normalize_lang,
    parse_model,
    require_api_key,
)
from proofread.exceptions import ConfigError, InputFileError


@pytest.fixture
def xml_file(tmp_path: Path) -> Path:
    """A tiny XML file in a temporary folder (pytest deletes it afterwards)."""
    path = tmp_path / "input.xml"
    path.write_text("<root><p>Hello.</p></root>", encoding="utf-8")
    return path


def make_settings(input_path: Path, **overrides):
    """Call build_settings() with sensible defaults; tests override what they need."""
    values = dict(
        input_path=input_path,
        lang="en",
        model=None,
        output_path=None,
        batch_size=10,
        concurrency=4,
        max_retries=3,
        timeout_s=60.0,
        strict=False,
    )
    values.update(overrides)
    return build_settings(**values)


# --- language ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "tag", "name"),
    [
        ("en", "en", "English"),
        ("fr", "fr", "French"),
        ("en-GB", "en-GB", "English (United Kingdom)"),
        ("EN_us", "en-US", "English (United States)"),
    ],
)
def test_normalize_lang_valid(raw, tag, name):
    assert normalize_lang(raw) == (tag, name)


@pytest.mark.parametrize("raw", ["", "   ", "english", "zz", "en-XX", "123"])
def test_normalize_lang_invalid(raw):
    with pytest.raises(ConfigError):
        normalize_lang(raw)


# --- model ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("google_genai:gemini-3.6-flash", ("google_genai", "gemini-3.6-flash")),
        ("ollama:qwen3:8b", ("ollama", "qwen3:8b")),
    ],
)
def test_parse_model_valid(raw, expected):
    assert parse_model(raw) == expected


@pytest.mark.parametrize("raw", ["qwen3", ":qwen3", "ollama:", ""])
def test_parse_model_invalid(raw):
    with pytest.raises(ConfigError):
        parse_model(raw)


# --- API keys ---------------------------------------------------------------


def test_api_key_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        require_api_key("google_genai")


def test_api_key_alternative_name_accepted(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    require_api_key("google_genai")  # no error


def test_ollama_needs_no_key():
    require_api_key("ollama")  # no error


def test_unknown_provider_skipped():
    require_api_key("some_future_provider")  # no error


# --- build_settings -----------------------------------------------------------


def test_default_output_path():
    assert default_output_path(Path("docs/input.xml")) == Path("docs/input.corrected.xml")


def test_defaults(xml_file, monkeypatch):
    monkeypatch.delenv("PROOFREAD_MODEL", raising=False)
    s = make_settings(xml_file)
    assert s.output_path == xml_file.with_name("input.corrected.xml")
    assert s.model == DEFAULT_MODEL


def test_model_from_env(xml_file, monkeypatch):
    monkeypatch.setenv("PROOFREAD_MODEL", "ollama:qwen3:8b")
    s = make_settings(xml_file)
    assert (s.provider, s.model_name) == ("ollama", "qwen3:8b")


def test_flag_beats_env(xml_file, monkeypatch):
    monkeypatch.setenv("PROOFREAD_MODEL", "ollama:qwen3:8b")
    assert make_settings(xml_file, model="google_genai:x").provider == "google_genai"


def test_missing_input_file(tmp_path):
    with pytest.raises(InputFileError):
        make_settings(tmp_path / "missing.xml")


def test_output_cannot_be_input(xml_file):
    with pytest.raises(ConfigError):
        make_settings(xml_file, output_path=xml_file)


@pytest.mark.parametrize(
    "override",
    [{"batch_size": 0}, {"concurrency": 0}, {"max_retries": -1}, {"timeout_s": 0}],
)
def test_bad_numbers(xml_file, override):
    with pytest.raises(ConfigError):
        make_settings(xml_file, **override)


# --- requests per minute ------------------------------------------------------------


def test_rpm_defaults_to_no_limit(xml_file, monkeypatch):
    monkeypatch.delenv("PROOFREAD_RPM", raising=False)
    assert make_settings(xml_file).requests_per_minute is None


def test_reasons_follow_env(xml_file, monkeypatch):
    monkeypatch.delenv("PROOFREAD_REASONS", raising=False)
    assert make_settings(xml_file).include_reasons is False
    monkeypatch.setenv("PROOFREAD_REASONS", "true")
    assert make_settings(xml_file).include_reasons is True
    monkeypatch.setenv("PROOFREAD_REASONS", "false")
    assert make_settings(xml_file).include_reasons is False
    assert make_settings(xml_file, include_reasons=False).include_reasons is False
    monkeypatch.setenv("PROOFREAD_REASONS", "false")
    assert make_settings(xml_file, include_reasons=True).include_reasons is True


def test_rpm_from_flag_and_env(xml_file, monkeypatch):
    monkeypatch.setenv("PROOFREAD_RPM", "15")
    assert make_settings(xml_file).requests_per_minute == 15
    assert make_settings(xml_file, requests_per_minute=30).requests_per_minute == 30


@pytest.mark.parametrize("bad", ["0", "-5", "fast"])
def test_bad_rpm_is_rejected(xml_file, monkeypatch, bad):
    monkeypatch.setenv("PROOFREAD_RPM", bad)
    with pytest.raises(ConfigError):
        make_settings(xml_file)
