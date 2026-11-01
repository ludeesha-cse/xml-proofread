"""Create the chat model from settings. The only place that knows about providers.

Everything else in the app talks to the returned Runnable and never needs to know
whether it is Gemini in the cloud or Ollama on this machine.
"""

import logging
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelError, OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from proofread.config import Settings
from proofread.exceptions import ConfigError
from proofread.schema import ProofreadResponse

logger = logging.getLogger(__name__)

# Ollama's default context window is small. If the prompt is longer, Ollama silently
# drops the start of it (our instructions!). 8192 tokens fits a batch of 10 paragraphs.
OLLAMA_CONTEXT_TOKENS = 8192

# LangChain package each provider needs, for a helpful error message.
PROVIDER_PACKAGES = {"google_genai": "langchain-google-genai", "ollama": "langchain-ollama"}


def model_kwargs(settings: Settings) -> dict[str, Any]:
    """Provider-specific options.

    Careful: chat models silently ignore options they don't know, so each provider
    gets only the names it actually supports.
    """
    kwargs: dict[str, Any] = {}
    if settings.provider != "google_genai":
        # Same input -> same output. Gemini 3 models ignore temperature (they use fixed
        # sampling defaults and LangChain warns about it), so it is not sent to Gemini.
        kwargs["temperature"] = 0
    if settings.provider == "ollama":
        kwargs["num_ctx"] = OLLAMA_CONTEXT_TOKENS
        kwargs["reasoning"] = False  # no "thinking" step: much faster, not needed here
        kwargs["client_kwargs"] = {"timeout": settings.timeout_s}
    elif settings.provider == "google_genai":
        kwargs["timeout"] = settings.timeout_s
        kwargs["max_retries"] = 0  # we retry ourselves (see build_structured_llm)
        kwargs["thinking_level"] = "minimal"
    return kwargs


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build the LangChain chat model for settings.provider / settings.model_name."""
    try:
        return init_chat_model(
            model=settings.model_name,
            model_provider=settings.provider,
            **model_kwargs(settings),
        )
    except ImportError as exc:
        package = PROVIDER_PACKAGES.get(
            settings.provider, f"the LangChain package for '{settings.provider}'"
        )
        raise ConfigError(
            f"Provider '{settings.provider}' needs {package}. Install it with: uv add {package}"
        ) from exc
    except ValueError as exc:  # e.g. a provider name LangChain doesn't know
        raise ConfigError(f"Cannot create model '{settings.model}': {exc}") from exc


def is_retryable(exc: BaseException) -> bool:
    """True if trying the same request again might succeed.

    Retry: rate limits (429), server errors (5xx), timeouts, connection problems and
    invalid model output. Don't retry: wrong model name (404), bad API key (401/403) or
    a rejected request (400), which fail the same way every time.
    """
    # 1. LangChain's standard errors say it themselves (Gemini raises these).
    if isinstance(exc, ModelError):
        return exc.is_retryable
    # 2. Errors that carry an HTTP status code (e.g. Ollama's ResponseError).
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status > 0:
        return status in (408, 429) or status >= 500
    # 3. Network trouble, or output that didn't match our schema.
    return isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            httpx.TransportError,  # covers connect errors and timeouts
            OutputParserException,
            ValidationError,
        ),
    )


def build_structured_llm(model: BaseChatModel, settings: Settings) -> Runnable:
    """Wrap the model so it returns a ProofreadResponse, with retries.

    with_structured_output: the model must answer in our schema (JSON schema mode).
    with_retry: if the error is one that can go away (see is_retryable), wait a
    little, with growing delays, and try again, up to max_retries extra times.
    """
    structured = model.with_structured_output(ProofreadResponse, method="json_schema")
    if settings.max_retries == 0:
        return structured
    return structured.with_retry(
        retry_if_exception_type=is_retryable,
        stop_after_attempt=settings.max_retries + 1,
    )
