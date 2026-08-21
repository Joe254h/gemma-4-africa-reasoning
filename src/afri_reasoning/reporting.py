from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _percent(value: float | None) -> str:
    return "Pending" if value is None else f"{100 * value:.1f}%"


def render_report(
    path: str | Path,
    automatic: dict[str, Any],
    human: dict[str, Any] | None,
    run_id: str,
) -> None:
    lines = [
        "# Gemma 4 four-language zero-shot baseline",
        "",
        f"- Run ID: `{run_id}`",
        f"- Report generated: `{datetime.now(UTC).isoformat()}`",
        "- Status: provisional until required native-language review is complete",
        (
            "- Run completeness: "
            f"{automatic.get('run_completeness', {}).get('completed_samples', 0)}/"
            f"{automatic.get('run_completeness', {}).get('requested_samples', 0)} generations"
        ),
        "",
        "## Automatic results",
        "",
        (
            "| Language | Samples | Final accuracy | Valid think/final format | "
            "Thinking present | Mean tok/s |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language, metrics in automatic["by_language"].items():
        lines.append(
            "| {language} | {samples} | {accuracy} | {format_rate} | {thinking_rate} | "
            "{tokens:.2f} |".format(
                language=language,
                samples=metrics["samples"],
                accuracy=_percent(metrics["final_answer_accuracy"]),
                format_rate=_percent(metrics["format_valid_rate"]),
                thinking_rate=_percent(metrics["thinking_present_rate"]),
                tokens=metrics["mean_tokens_per_second"] or 0.0,
            )
        )

    lines.extend(
        [
            "",
            (
                "Automatic numeric accuracy is not a substitute for checking the logic "
                "of the reasoning."
            ),
            "",
            "## Human-reviewed results",
            "",
        ]
    )
    if human is None:
        lines.append(
            "Pending. Complete `human_review.csv`, then rerun the score command with `--reviews`."
        )
    else:
        lines.extend(
            [
                "| Language | Verified rows | Reasoning correct | Thinking language | "
                "Answer language | Answer English fallback |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for language, metrics in human["by_language"].items():
            lines.append(
                "| {language} | {verified}/{rows} | {reasoning} | {thinking_lang} | "
                "{answer_lang} | {fallback} |".format(
                    language=language,
                    verified=metrics["verified_rows"],
                    rows=metrics["rows"],
                    reasoning=_percent(metrics["reasoning_correct_rate"]),
                    thinking_lang=_percent(metrics["thinking_language_compliance_rate"]),
                    answer_lang=_percent(metrics["answer_language_compliance_rate"]),
                    fallback=_percent(metrics["answer_english_fallback_rate"]),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Compare languages on matched `problem_id` values.",
            (
                "- Do not call Swahili or Wolof results verified when their native-review "
                "count is incomplete."
            ),
            "- Do not compare this run with a fine-tuned run unless model revision, prompts, "
            "selected rows, "
            "and decoding settings are held constant.",
            "- Inspect parse failures and English fallback examples before reporting an average.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")
