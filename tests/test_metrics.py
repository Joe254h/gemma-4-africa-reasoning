from decimal import Decimal

from afri_reasoning.metrics import (
    automatic_metrics,
    extract_last_number,
    normalize_number,
    prediction_is_correct,
)


def test_extracts_last_number_from_reasoned_answer() -> None:
    text = "We first calculate 12 + 8 = 20. Final answer: 20"
    assert extract_last_number(text) == "20"


def test_normalizes_common_number_formats() -> None:
    assert normalize_number("1,234") == Decimal("1234")
    assert normalize_number("12,5") == Decimal("12.5")
    assert normalize_number("1 234") == Decimal("1234")


def test_numeric_correctness_uses_final_number() -> None:
    correct, predicted = prediction_is_correct("Steps mention 4. Jibu la mwisho: 9", "9")
    assert correct is True
    assert predicted == "9"


def test_automatic_metrics_keep_reasoning_human_scored() -> None:
    result = automatic_metrics(
        [
            {
                "sample_id": "p1:sw",
                "language": "sw",
                "content": "Jibu la mwisho: 7",
                "thinking": "Hatua moja...",
                "expected_answer": "7",
                "parse_valid": True,
                "latency_seconds": 2.0,
                "tokens_per_second": 10.0,
            }
        ]
    )
    assert result["by_language"]["sw"]["final_answer_accuracy"] == 1.0
    assert result["scope"] == "automatic-only"
    assert "human review" in result["warning"].lower()
