"""Command-line entry point."""

import logging
from pathlib import Path
from typing import Annotated

import typer
from dotenv import find_dotenv, load_dotenv

from proofread import __version__
from proofread.config import DEFAULT_MODEL, MODEL_ENV_VAR, build_settings, require_api_key
from proofread.exceptions import ProofreadError
from proofread.logging_setup import setup_logging

logger = logging.getLogger("proofread")

app = typer.Typer(
    add_completion=False,  # hide Typer's shell-completion options
    help="Proofread <p> elements in an XML file and write <input>.corrected.xml.",
)


def _version_callback(value: bool) -> None:
    """Runs as soon as --version is seen: print the version and stop."""
    if value:
        typer.echo(f"proofread {__version__}")
        raise typer.Exit()


@app.command()
def run(
    # Annotated[type, typer.Argument/Option(...)] = default
    # The type tells Typer how to convert the text; the default makes it optional.
    input_path: Annotated[Path, typer.Argument(help="XML file to proofread.", show_default=False)],
    lang: Annotated[
        str, typer.Option("--lang", "-l", help="BCP-47 language tag, e.g. en, en-GB, fr.")
    ] = "en",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=f"provider:model. Falls back to ${MODEL_ENV_VAR}, then {DEFAULT_MODEL}.",
            show_default=False,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path. Default: <input>.corrected.xml."),
    ] = None,
    batch_size: Annotated[int, typer.Option(help="<p> tags per LLM request.")] = 10,
    concurrency: Annotated[int, typer.Option(help="Parallel LLM requests.")] = 4,
    max_retries: Annotated[int, typer.Option(help="Retries per failed LLM request.")] = 3,
    timeout: Annotated[float, typer.Option(help="Seconds before an LLM request times out.")] = 60.0,
    strict: Annotated[
        bool, typer.Option(help="Fail the run if any paragraph cannot be proofread.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,  # handle --version before checking other options
            help="Show the version and exit.",
        ),
    ] = None,
) -> None:
    """Proofread INPUT_PATH and write the annotated copy next to it."""
    setup_logging(verbose)

    # Read .env from the folder the command is run in.
    # override=False -> variables already set in the shell win over the .env file.
    load_dotenv(find_dotenv(usecwd=True), override=False)

    try:
        settings = build_settings(
            input_path=input_path,
            lang=lang,
            model=model,
            output_path=output,
            batch_size=batch_size,
            concurrency=concurrency,
            max_retries=max_retries,
            timeout_s=timeout,
            strict=strict,
        )
        require_api_key(settings.provider)

        logger.info("Input:    %s", settings.input_path)
        logger.info("Output:   %s", settings.output_path)
        logger.info("Language: %s (%s)", settings.lang, settings.language_name)
        logger.info("Model:    %s", settings.model)
        logger.debug("Settings: %s", settings)

        # TODO: parse XML -> proofread -> inject -> write.
        logger.warning("Proofreading pipeline is not implemented yet")

    except ProofreadError as exc:
        # Expected problem (the user can fix it): short message, specific exit code.
        logger.error("%s", exc)
        raise typer.Exit(code=exc.exit_code) from None
    except KeyboardInterrupt:
        # Ctrl+C
        logger.error("Interrupted.")
        raise typer.Exit(code=130) from None
    except Exception:
        # A real bug. logger.exception() also prints the stack trace.
        logger.exception("Unexpected error. Please report this with the log above.")
        raise typer.Exit(code=1) from None


def main() -> None:
    """Entry point for the `proofread` command (see [project.scripts] in pyproject.toml)."""
    app()
