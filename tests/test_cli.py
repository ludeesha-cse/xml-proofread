"""Tests for cli.py: run the command in-process and check exit codes."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from proofread.cli import app

runner = CliRunner()


@pytest.fixture
def xml_file(tmp_path: Path) -> Path:
    path = tmp_path / "input.xml"
    path.write_text("<root><p>Hello.</p></root>", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)  # autouse = applies to every test in this file
def clean_env(monkeypatch, tmp_path):
    """Make every test independent of your real shell and .env file."""
    monkeypatch.chdir(tmp_path)  # no .env in this folder, so yours isn't loaded
    monkeypatch.delenv("PROOFREAD_MODEL", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key-for-tests")


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "proofread" in result.output


def test_valid_run(xml_file):
    result = runner.invoke(app, [str(xml_file), "--lang", "fr"])
    assert result.exit_code == 0


def test_ollama_run_without_key(xml_file, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    result = runner.invoke(app, [str(xml_file), "--model", "ollama:qwen3:8b"])
    assert result.exit_code == 0


def test_missing_file(tmp_path):
    assert runner.invoke(app, [str(tmp_path / "missing.xml")]).exit_code == 3


def test_bad_lang(xml_file):
    assert runner.invoke(app, [str(xml_file), "--lang", "en-XX"]).exit_code == 2


def test_bad_model(xml_file):
    assert runner.invoke(app, [str(xml_file), "--model", "qwen3"]).exit_code == 2


def test_missing_api_key(xml_file, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert runner.invoke(app, [str(xml_file)]).exit_code == 2
