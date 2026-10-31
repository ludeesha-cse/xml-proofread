"""Settings for one run of the tool.

All command-line values are checked here, once, before any real work starts.
The result is a frozen Settings object that the rest of the program reads.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import langcodes

from proofread.exceptions import ConfigError, InputFileError

logger = logging.getLogger(__name__)

# The model is written as "provider:model_name".
#   Cloud: google_genai:gemini-3.5-flash-lite
#   Local: ollama:qwen3:8b
DEFAULT_MODEL = "google_genai:gemini-3.5-flash-lite"
MODEL_ENV_VAR = "PROOFREAD_MODEL"  # lets .env choose the model without a flag

OUTPUT_SUFFIX = ".corrected.xml"

# Which environment variable holds each provider's API key.
# Any one of the names in the tuple is enough.
# An empty tuple means "no key needed" (Ollama runs on your own machine).
# Providers not listed here are still allowed; LangChain reports missing keys for them.
PROVIDER_API_KEYS: dict[str, tuple[str, ...]] = {
    "google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "ollama": (),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

_LANG_HINT = "Use a BCP-47 tag such as 'en', 'en-GB' or 'fr'."


# Immutable Value Object / Parameter Object.
@dataclass(frozen=True)  # frozen = cannot be changed after it is created
class Settings:
    """Everything one run needs. Created only by build_settings()."""

    input_path: Path
    output_path: Path
    lang: str  # tag, e.g. "en-GB"
    language_name: str  # e.g. "English (United Kingdom), formal/readable for prompts"
    provider: str  # e.g. "ollama"
    model_name: str  # e.g. "qwen3:8b"
    batch_size: int  # <p> tags per LLM request
    concurrency: int  # LLM requests running at the same time
    max_retries: int
    timeout_s: float  # seconds before one request gives up
    strict: bool  # does we fail the whole run if any paragraph fails

    @property
    def model(self) -> str:
        """The full model string again, for logs: "ollama:qwen3:8b"."""
        return f"{self.provider}:{self.model_name}"


def default_output_path(input_path: Path) -> Path:
    """docs/input.xml -> docs/input.corrected.xml (same folder as the input)."""
    return input_path.with_name(input_path.stem + OUTPUT_SUFFIX)


def normalize_lang(tag: str) -> tuple[str, str]:
    """Check a BCP-47 tag. Returns (normalised tag, readable name).

    "EN_us" -> ("en-US", "English (United States)")
    """
    raw = tag.strip()
    if not raw:
        raise ConfigError(f"--lang must not be empty. {_LANG_HINT}")

    try:
        normalized = langcodes.standardize_tag(raw)  # fixes case and "_" -> "-"
    except (langcodes.LanguageTagError, ValueError) as exc:
        # "from exc" keeps the original error attached, which helps debugging.
        raise ConfigError(f"--lang '{tag}' is not a valid language tag. {_LANG_HINT}") from exc

    if not langcodes.tag_is_valid(normalized):
        raise ConfigError(f"--lang '{tag}' is not a valid language tag. {_LANG_HINT}")

    name = langcodes.Language.get(normalized).display_name()
    # tag_is_valid() accepts tags like "en-XX", whose name is
    # "English (Unknown Region [XX])". The model needs a real language, so reject them.
    if "Unknown" in name:
        raise ConfigError(f"--lang '{tag}' is not a known language. {_LANG_HINT}")

    return normalized, name


def parse_model(model: str) -> tuple[str, str]:
    """Split "provider:model_name" on the FIRST colon only.

    "ollama:qwen3:8b" -> ("ollama", "qwen3:8b")
    """
    provider, sep, name = model.strip().partition(":")
    if not sep or not provider or not name:
        raise ConfigError(
            f"--model '{model}' must look like 'provider:model_name', "
            f"e.g. '{DEFAULT_MODEL}' or 'ollama:qwen3:8b'."
        )
    return provider, name


def require_api_key(provider: str) -> None:
    """Fail early, with a clear message, if the provider's API key is missing."""
    names = PROVIDER_API_KEYS.get(provider)
    if names is None:
        logger.debug("Unknown provider '%s'; skipping API key check.", provider)
        return
    if not names:
        return  # provider needs no key (Ollama)
    if not any(os.environ.get(name) for name in names):
        raise ConfigError(
            f"No API key found for provider '{provider}'. "
            f"Set {' or '.join(names)} in your environment or in .env (see .env.example)."
        )


def build_settings(
    *,  # the "*" forces keyword arguments, so values can't be passed in the wrong order
    input_path: Path,
    lang: str,
    model: str | None,
    output_path: Path | None,
    batch_size: int,
    concurrency: int,
    max_retries: int,
    timeout_s: float,
    strict: bool,
) -> Settings:
    """Check raw CLI values and return a Settings object, or raise a ProofreadError."""
    # Factory + Fail-Fast validation.
    # 1. The input file must exist and be a file (not a folder).
    if not input_path.exists():
        raise InputFileError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise InputFileError(f"Input path is not a file: {input_path}")

    # 2. Numbers must make sense.
    if batch_size < 1:
        raise ConfigError("--batch-size must be at least 1.")
    if concurrency < 1:
        raise ConfigError("--concurrency must be at least 1.")
    if max_retries < 0:
        raise ConfigError("--max-retries must be 0 or more.")
    if timeout_s <= 0:
        raise ConfigError("--timeout must be greater than 0.")

    # 3. Language.
    lang_tag, language_name = normalize_lang(lang)

    # 4. Model: --model flag first, then PROOFREAD_MODEL, then the default.
    provider, model_name = parse_model(model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL)

    # 5. Never overwrite the input file.
    resolved_output = output_path or default_output_path(input_path)
    if resolved_output.resolve() == input_path.resolve():
        raise ConfigError("Output path must be different from the input path.")

    return Settings(
        input_path=input_path,
        output_path=resolved_output,
        lang=lang_tag,
        language_name=language_name,
        provider=provider,
        model_name=model_name,
        batch_size=batch_size,
        concurrency=concurrency,
        max_retries=max_retries,
        timeout_s=timeout_s,
        strict=strict,
    )
