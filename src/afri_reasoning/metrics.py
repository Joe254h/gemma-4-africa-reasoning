from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any

NUMBER_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,]\d+)?")


def normalize_number(raw: str | int | float) -> Decimal | None:
    value = str(raw).strip().replace(" ", "")
    if not value:
        return None
    if "," in value and "." not in value:
        left, right = value.rsplit(",", 1)
        value = left + right if len(right) == 3 else left + "." + right
    else:
        value = value.replace(",", "")
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text or "")
    return matches[-1] if matches else None


def prediction_is_correct(content: str, expected_answer: str) -> tuple[bool, str | None]:
    predicted_raw = extract_last_number(content)
    predicted = normalize_number(predicted_raw) if predicted_raw is not None else None
    expected = normalize_number(expected_answer)
    return predicted is not None and expected is not None and predicted == expected, predicted_raw


def automatic_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotated: list[dict[str, Any]] = []
    for prediction in predictions:
        correct, predicted_answer = prediction_is_correct(
            prediction.get("content", ""), str(prediction.get("expected_answer", ""))
        )
        format_valid = bool(
            prediction.get("parse_valid")
            and prediction.get("thinking", "").strip()
            and prediction.get("content", "").strip()
        )
        row = {
            **prediction,
            "predicted_answer": predicted_answer,
            "final_answer_correct_auto": correct,
            "format_valid_auto": format_valid,
            "thinking_present_auto": bool(prediction.get("thinking", "").strip()),
        }
        annotated.append(row)
        by_language[row["language"]].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        return {
            "samples": count,
            "final_answer_accuracy": (
                sum(row["final_answer_correct_auto"] for row in rows) / count if count else None
            ),
            "format_valid_rate": (
                sum(row["format_valid_auto"] for row in rows) / count if count else None
            ),
            "thinking_present_rate": (
                sum(row["thinking_present_auto"] for row in rows) / count if count else None
            ),
            "mean_latency_seconds": (
                sum(float(row.get("latency_seconds", 0)) for row in rows) / count
                if count
                else None
            ),
            "mean_tokens_per_second": (
                sum(float(row.get("tokens_per_second", 0)) for row in rows) / count
                if count
                else None
            ),
        }

    return {
        "scope": "automatic-only",
        "warning": (
            "Reasoning correctness and language compliance require the human review sheet."
        ),
        "overall": summarize(annotated),
        "by_language": {
            language: summarize(rows) for language, rows in sorted(by_language.items())
        },
        "annotated_predictions": annotated,
    }
