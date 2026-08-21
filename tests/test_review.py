import csv
from pathlib import Path

from afri_reasoning.review import REVIEW_COLUMNS, aggregate_human_reviews


def _write_review(path: Path, **updates: str) -> None:
    row = {column: "" for column in REVIEW_COLUMNS}
    row.update(
        {
            "sample_id": "p1:wo",
            "language": "wo",
            "reasoning_correct": "true",
            "thinking_language_correct": "true",
            "answer_language_correct": "true",
            "thinking_fallback_to_english": "false",
            "answer_fallback_to_english": "false",
            "reviewer": "Native Reviewer",
            "reviewer_native_language": "wo",
            "review_date": "2026-08-13",
        }
    )
    row.update(updates)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def test_native_wolof_review_counts_as_verified(tmp_path: Path) -> None:
    path = tmp_path / "reviews.csv"
    _write_review(path)
    result = aggregate_human_reviews(path)
    assert result["by_language"]["wo"]["verified_rows"] == 1
    assert result["by_language"]["wo"]["reasoning_correct_rate"] == 1.0


def test_non_native_wolof_review_is_not_verified(tmp_path: Path) -> None:
    path = tmp_path / "reviews.csv"
    _write_review(path, reviewer_native_language="en")
    result = aggregate_human_reviews(path)
    assert result["by_language"]["wo"]["verified_rows"] == 0
    assert result["by_language"]["wo"]["reasoning_correct_rate"] is None
