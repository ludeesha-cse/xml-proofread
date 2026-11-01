"""Create the chat model from settings. The only place that knows about providers.

Everything else in the app talks to the returned Runnable and never needs to know
whether it is Gemini in the cloud or Ollama on this machine.
"""

import logging
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelError, ModelRateLimitError, OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from proofread.config import Settings
from proofread.exceptions import ConfigError
from proofread.schema import ProofreadResponse

logger = logging.getLogger(__name__)

# Ollama's context window must fit the whole prompt plus the answer. If it is too
# small, Ollama silently drops the START of the prompt, which is our instructions.
# Measured with qwen3: the instructions take about 1,700 tokens, and each 50-word
# paragraph about 530 more (its text and token list, plus room for the answer).
OLLAMA_BASE_TOKENS = 2048
OLLAMA_TOKENS_PER_PARAGRAPH = 600  # a little above the measured 530, for safety
OLLAMA_MIN_CONTEXT = 8192
OLLAMA_MAX_CONTEXT = 32768  # bigger windows need a lot more memory on a 16 GB Mac

# How to install each provider's LangChain package, for a helpful error message.
# Gemini and Ollama are core dependencies; OpenAI and Anthropic are optional extras.
PROVIDER_INSTALL = {
    "google_genai": "uv add langchain-google-genai",
    "ollama": "uv add langchain-ollama",
    "openai": "uv sync --extra openai",
    "anthropic": "uv sync --extra anthropic",
}

# Providers whose models accept temperature=0. Gemini 3 uses fixed sampling and warns,
# and newer Claude models reject it ("omit temperature"). LangChain's OpenAI class
# already drops it for GPT-5 models that don't support it.
TEMPERATURE_PROVIDERS = {"ollama", "openai"}


def ollama_context_size(batch_size: int) -> int:
    """Context window (num_ctx) for a batch, rounded up to a multiple of 1024."""
    needed = OLLAMA_BASE_TOKENS + OLLAMA_TOKENS_PER_PARAGRAPH * batch_size
    rounded = -(-needed // 1024) * 1024  # round up: -(-a // b) is ceiling division
    if rounded > OLLAMA_MAX_CONTEXT:
        logger.warning(
            "--batch-size %d is too large for Ollama; long batches may be cut off. Use 50 or less.",
            batch_size,
        )
    return min(max(rounded, OLLAMA_MIN_CONTEXT), OLLAMA_MAX_CONTEXT)


def model_kwargs(settings: Settings) -> dict[str, Any]:
    """Provider-specific options.

    Careful: chat models silently ignore options they don't know, so each provider
    gets only the names it actually supports.
    """
    kwargs: dict[str, Any] = {}
    if settings.provider in TEMPERATURE_PROVIDERS:
        kwargs["temperature"] = 0  # same input -> same output
    if settings.provider == "ollama":
        kwargs["num_ctx"] = ollama_context_size(settings.batch_size)
        kwargs["reasoning"] = False  # no "thinking" step: much faster, not needed here
        kwargs["client_kwargs"] = {"timeout": settings.timeout_s}
    elif settings.provider in ("google_genai", "openai", "anthropic"):
        kwargs["timeout"] = settings.timeout_s
        kwargs["max_retries"] = 0  # we retry ourselves (see build_structured_llm)
    if settings.requests_per_minute:
        # Every chat model accepts a rate limiter. Each request (and each retry) waits
        # for a slot, so we stay under the provider's limit instead of being refused.
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=settings.requests_per_minute / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=1,  # no bursts: spread requests evenly
        )
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
        install = PROVIDER_INSTALL.get(settings.provider)
        hint = f" Install it with: {install}" if install else ""
        raise ConfigError(
            f"Provider '{settings.provider}' needs its LangChain package.{hint}"
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


def is_rate_limited(exc: BaseException) -> bool:
    """True if the provider refused the request because we sent too many (HTTP 429)."""
    return isinstance(exc, ModelRateLimitError) or getattr(exc, "status_code", None) == 429


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
        # Waits of about 2, 4, 8... seconds (plus up to 1 s of random jitter), at most 60.
        exponential_jitter_params={"initial": 2, "max": 60},
    )
