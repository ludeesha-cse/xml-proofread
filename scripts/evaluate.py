"""Compare models on a sample file: accuracy against an expected output, and speed.

Usage:
    uv run python scripts/evaluate.py samples/input.xml samples/expected_output.xml \\
        -m google_genai:gemini-3.5-flash-lite -m google_genai:gemini-3.6-flash --runs 3

Each model runs `--runs` times (results vary between runs), and the averages are
printed as a Markdown table you can paste into the README.
"""

import logging
import statistics
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from proofread.config import build_settings, require_api_key
from proofread.evaluation import Score, corrected_text, score
from proofread.exceptions import ProofreadError
from proofread.pipeline import RunResult, run
from proofread.xml_io import find_paragraphs, load_xml


def run_once(input_path: Path, model: str, lang: str, out_dir: Path) -> RunResult:
    settings = build_settings(
        input_path=input_path, lang=lang, model=model, output_path=out_dir / "out.xml",
        batch_size=10, concurrency=4, max_retries=2, timeout_s=180.0, strict=False,
    )  # fmt: skip
    require_api_key(settings.provider)
    return run(settings)


def print_differences(reference_path: Path, output_path: Path) -> None:
    """Show each paragraph where the model's corrected text differs from the reference."""
    reference = find_paragraphs(load_xml(reference_path).root)
    output = find_paragraphs(load_xml(output_path).root)
    for ref, out in zip(reference, output, strict=True):
        expected, actual = corrected_text(ref), corrected_text(out)
        if expected != actual:
            typer.echo(f"    expected: {expected}\n    model:    {actual}")


def main(
    input_path: Path,
    expected_path: Path,
    models: Annotated[list[str], typer.Option("--model", "-m", help="provider:model")],
    runs: Annotated[int, typer.Option(help="Runs per model.")] = 3,
    lang: Annotated[str, typer.Option(help="BCP-47 language tag.")] = "en",
    show_differences: Annotated[bool, typer.Option(help="Print mismatching text.")] = False,
) -> None:
    load_dotenv(override=False)
    logging.basicConfig(level=logging.WARNING)  # keep the per-run INFO logs quiet
    reference = load_xml(expected_path).root

    rows = []
    for model in models:
        scores: list[Score] = []
        results: list[RunResult] = []
        for attempt in range(1, runs + 1):
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    result = run_once(input_path, model, lang, Path(tmp))
                except ProofreadError as exc:
                    typer.echo(f"{model} run {attempt}: failed: {exc}", err=True)
                    continue
                scores.append(score(reference, load_xml(result.output_path).root))
                results.append(result)
                typer.echo(f"{model} run {attempt}: F1 {scores[-1].f1:.2f}, "
                           f"{result.total_seconds:.1f}s", err=True)  # fmt: skip
                if show_differences:
                    print_differences(expected_path, result.output_path)
        if not scores:
            continue

        def mean(values) -> float:
            return statistics.mean(values)

        rows.append(
            f"| {model} | {len(scores)} "
            f"| {mean(s.precision for s in scores):.2f} "
            f"| {mean(s.recall for s in scores):.2f} "
            f"| {mean(s.f1 for s in scores):.2f} "
            f"| {mean(s.exact for s in scores):.1f}/{scores[0].paragraphs} "
            f"| {mean(r.total_seconds for r in results):.1f}s "
            f"| {mean(r.input_tokens for r in results):,.0f} / "
            f"{mean(r.output_tokens for r in results):,.0f} |"
        )

    typer.echo("\n| Model | Runs | Precision | Recall | F1 | Exact paragraphs | Avg time "
               "| Avg tokens (in / out) |")  # fmt: skip
    typer.echo("|---|---|---|---|---|---|---|---|")
    typer.echo("\n".join(rows))


if __name__ == "__main__":
    typer.run(main)
