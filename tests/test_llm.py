"""Tests for llm.py. Creating a model makes no network calls, so these run offline."""

from pathlib import Path

import httpx
import ollama
import pytest
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelNotFoundError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.runnables import RunnableLambda

from proofread.config import build_settings
from proofread.exceptions import ConfigError
from proofread.llm import (
    OLLAMA_MAX_CONTEXT,
    OLLAMA_MIN_CONTEXT,
    ollama_context_size,
    build_structured_llm,
    create_chat_model,
    is_retryable,
    model_kwargs,
)


@pytest.fixture
def settings_for(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")  # Gemini's class needs some key
    xml = tmp_path / "in.xml"
    xml.write_text("<r/>")

    def make(model: str, **overrides):
        values = dict(
            input_path=xml, lang="en", model=model, output_path=None, batch_size=10,
            concurrency=1, max_retries=3, timeout_s=42.0, strict=False,
        )  # fmt: skip
        values.update(overrides)
        return build_settings(**values)

    return make


def test_ollama_model_settings(settings_for):
    model = create_chat_model(settings_for("ollama:qwen3:8b"))
    assert type(model).__name__ == "ChatOllama"
    assert model.model == "qwen3:8b"
    assert model.temperature == 0
    assert model.num_ctx == ollama_context_size(10)
    assert model.reasoning is False
    assert model.client_kwargs == {"timeout": 42.0}


def test_gemini_model_settings(settings_for):
    model = create_chat_model(settings_for("google_genai:gemini-3.6-flash"))
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert "temperature" not in model_kwargs(settings_for("google_genai:gemini-3.6-flash"))
    assert model.timeout == 42.0
    assert model.max_retries == 0


def test_openai_model_settings(settings_for, monkeypatch):
    pytest.importorskip("langchain_openai")  # optional extra: skip if not installed
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    model = create_chat_model(settings_for("openai:gpt-5.4-mini"))
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "gpt-5.4-mini"
    assert model.request_timeout == 42.0
    assert model.max_retries == 0
    assert model_kwargs(settings_for("openai:gpt-5.4-mini"))["temperature"] == 0


def test_anthropic_model_settings(settings_for, monkeypatch):
    pytest.importorskip("langchain_anthropic")  # optional extra: skip if not installed
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    model = create_chat_model(settings_for("anthropic:claude-haiku-5-5"))
    assert type(model).__name__ == "ChatAnthropic"
    assert model.model == "claude-haiku-5-5"
    assert model.default_request_timeout == 42.0
    assert model.max_retries == 0
    # Newer Claude models reject temperature, so it must not be sent.
    assert "temperature" not in model_kwargs(settings_for("anthropic:claude-haiku-5-5"))


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (1, OLLAMA_MIN_CONTEXT),  # small batches still get the minimum
        (10, OLLAMA_MIN_CONTEXT),  # 2048 + 6000 = 8048 -> 8192
        (20, 14336),  # 2048 + 12000 = 14048 -> rounded up to 14 x 1024
        (100, OLLAMA_MAX_CONTEXT),  # capped
    ],
)
def test_ollama_context_grows_with_batch_size(batch_size, expected):
    assert ollama_context_size(batch_size) == expected


def test_too_large_batch_for_ollama_warns(caplog):
    ollama_context_size(100)
    assert "too large for Ollama" in caplog.text

    
def test_unknown_provider_gives_config_error(settings_for):
    with pytest.raises(ConfigError):
        create_chat_model(settings_for("no_such_provider:model"))


def test_structured_llm_is_a_runnable(settings_for):
    settings = settings_for("ollama:qwen3:8b")
    llm = build_structured_llm(create_chat_model(settings), settings)
    assert hasattr(llm, "invoke")


def test_missing_provider_package_gives_install_hint(settings_for, monkeypatch):
    def missing_package(**kwargs):
        raise ImportError("No module named 'langchain_ollama'")

    monkeypatch.setattr("proofread.llm.init_chat_model", missing_package)
    with pytest.raises(ConfigError, match="uv add langchain-ollama"):
        create_chat_model(settings_for("ollama:qwen3:8b"))
    with pytest.raises(ConfigError, match="uv sync --extra openai"):
        create_chat_model(settings_for("openai:gpt-5.4-mini"))


# --- is_retryable -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ModelRateLimitError("429"), True),
        (ModelTimeoutError("slow"), True),
        (ModelNotFoundError("404"), False),
        (ModelAuthenticationError("401"), False),
        (ollama.ResponseError("model 'x' not found", 404), False),
        (ollama.ResponseError("overloaded", 503), True),
        (httpx.ConnectError("refused"), True),
        (httpx.ReadTimeout("slow"), True),
        (TimeoutError(), True),
        (OutputParserException("bad json"), True),
        (ValueError("a bug"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected


def test_permanent_errors_are_not_retried(settings_for, monkeypatch):
    calls = []

    def always_404(_):
        calls.append(1)
        raise ModelNotFoundError("model not found")

    settings = settings_for("ollama:qwen3:8b", max_retries=3)
    fake_model = RunnableLambda(always_404)
    fake_model.with_structured_output = lambda *a, **k: fake_model  # skip the real wrapper
    llm = build_structured_llm(fake_model, settings)
    with pytest.raises(ModelNotFoundError):
        llm.invoke("hi")
    assert len(calls) == 1  # failed at once, no retries
