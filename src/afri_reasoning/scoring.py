from __future__ import annotations

from pathlib import Path
from typing import Any

from afri_reasoning.io import read_json, read_jsonl, write_json_atomic
from afri_reasoning.metrics import automatic_metrics
from afri_reasoning.reporting import render_report
from afri_reasoning.review import aggregate_human_reviews, write_review_queue


def score_run(run_dir: str | Path, reviews: str | Path | None = None) -> dict[str, Any]:
    directory = Path(run_dir).expanduser().resolve()
    predictions_path = directory / "predictions.jsonl"
    manifest_path = directory / "manifest.json"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    predictions = read_jsonl(predictions_path)
    sample_ids = [prediction.get("sample_id") for prediction in predictions]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id values invalidate this run")

    manifest = read_json(manifest_path)
    automatic = automatic_metrics(predictions)
    annotated = automatic.pop("annotated_predictions")
    requested = int(manifest.get("requested_samples", len(predictions)))
    automatic["run_completeness"] = {
        "manifest_status": manifest.get("status"),
        "requested_samples": requested,
        "completed_samples": len(predictions),
        "complete": manifest.get("status") == "completed" and len(predictions) == requested,
    }
    write_json_atomic(directory / "automatic_metrics.json", automatic)

    review_path = directory / "human_review.csv"
    if reviews is None:
        if not review_path.exists():
            write_review_queue(review_path, annotated)
        human = None
    else:
        review_path = Path(reviews).expanduser().resolve()
        human = aggregate_human_reviews(review_path)
        write_json_atomic(directory / "human_metrics.json", human)

    render_report(directory / "report.md", automatic, human, directory.name)
    return {
        "run_dir": str(directory),
        "automatic_metrics": str(directory / "automatic_metrics.json"),
        "human_review": str(review_path),
        "human_metrics": str(directory / "human_metrics.json") if human else None,
        "report": str(directory / "report.md"),
    }
