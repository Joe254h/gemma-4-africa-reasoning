from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

REVIEW_COLUMNS = [
    "sample_id",
    "language",
    "question",
    "thinking",
    "final_response",
    "expected_answer",
    "predicted_answer",
    "final_answer_correct_auto",
    "format_valid_auto",
    "reasoning_correct",
    "thinking_language_correct",
    "answer_language_correct",
    "thinking_fallback_to_english",
    "answer_fallback_to_english",
    "reviewer",
    "reviewer_native_language",
    "review_date",
    "notes",
]


def write_review_queue(path: str | Path, predictions: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "language": row["language"],
                    "question": row["question"],
                    "thinking": row.get("thinking", ""),
                    "final_response": row.get("content", ""),
                    "expected_answer": row.get("expected_answer", ""),
                    "predicted_answer": row.get("predicted_answer", ""),
                    "final_answer_correct_auto": row.get("final_answer_correct_auto", ""),
                    "format_valid_auto": row.get("format_valid_auto", ""),
                    "reasoning_correct": "",
                    "thinking_language_correct": "",
                    "answer_language_correct": "",
                    "thinking_fallback_to_english": "",
                    "answer_fallback_to_english": "",
                    "reviewer": "",
                    "reviewer_native_language": "",
                    "review_date": "",
                    "notes": "",
                }
            )
    temporary.replace(destination)


def _optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    if not normalized:
        return None
    raise ValueError(f"Expected true/false or blank, received {value!r}")


def aggregate_human_reviews(path: str | Path) -> dict[str, Any]:
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), start=2):
            row = dict(raw)
            for field in [
                "reasoning_correct",
                "thinking_language_correct",
                "answer_language_correct",
                "thinking_fallback_to_english",
                "answer_fallback_to_english",
            ]:
                try:
                    row[field] = _optional_bool(row.get(field, ""))
                except ValueError as exc:
                    raise ValueError(f"Review CSV line {line_number}: {exc}") from exc

            required_scores = [
                row["reasoning_correct"],
                row["thinking_language_correct"],
                row["answer_language_correct"],
            ]
            scored = all(value is not None for value in required_scores)
            reviewer_present = bool(row.get("reviewer", "").strip())
            review_date_present = bool(row.get("review_date", "").strip())
            native_ok = True
            if row["language"] in {"sw", "wo"}:
                native_language = row.get("reviewer_native_language", "").strip().lower()
                native_ok = native_language == row["language"]
            row["verified"] = scored and reviewer_present and review_date_present and native_ok
            by_language[row["language"]].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        verified = [row for row in rows if row["verified"]]
        count = len(verified)

        def rate(field: str) -> float | None:
            return sum(bool(row[field]) for row in verified) / count if count else None

        return {
            "rows": len(rows),
            "verified_rows": count,
            "reasoning_correct_rate": rate("reasoning_correct"),
            "thinking_language_compliance_rate": rate("thinking_language_correct"),
            "answer_language_compliance_rate": rate("answer_language_correct"),
            "thinking_english_fallback_rate": rate("thinking_fallback_to_english"),
            "answer_english_fallback_rate": rate("answer_fallback_to_english"),
        }

    return {
        "scope": "human-reviewed",
        "verification_rule": (
            "Swahili and Wolof rows require reviewer_native_language equal to sw or wo."
        ),
        "by_language": {
            language: summarize(rows) for language, rows in sorted(by_language.items())
        },
    }
