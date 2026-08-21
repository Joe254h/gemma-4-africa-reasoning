from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afri_reasoning.config import repo_path
from afri_reasoning.io import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from afri_reasoning.run import SHARDING_STRATEGY


def merge_baseline_shards(
    config: dict[str, Any], *, shard_dirs: list[str | Path], output_dir: str | Path
) -> Path:
    """Validate and merge completed baseline shards in evaluation-file order."""
    if len(shard_dirs) < 2:
        raise ValueError("At least two shard directories are required")

    evaluation_path = repo_path(config, config["paths"]["evaluation_file"])
    evaluation = read_jsonl(evaluation_path)
    expected_ids = [str(sample["sample_id"]) for sample in evaluation]
    expected_id_set = set(expected_ids)
    config_sha256 = sha256_file(config["_meta"]["config_path"])
    evaluation_sha256 = sha256_file(evaluation_path)

    manifests: list[dict[str, Any]] = []
    predictions_by_id: dict[str, dict[str, Any]] = {}
    shard_sources: list[dict[str, Any]] = []

    for raw_directory in shard_dirs:
        directory = Path(raw_directory).expanduser().resolve()
        manifest_path = directory / "manifest.json"
        predictions_path = directory / "predictions.jsonl"
        if not manifest_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"Incomplete shard directory: {directory}")

        manifest = read_json(manifest_path)
        predictions = read_jsonl(predictions_path)
        if manifest.get("status") != "completed":
            raise ValueError(f"Shard is not completed: {directory}")
        if manifest.get("configuration_sha256") != config_sha256:
            raise ValueError(f"Configuration checksum mismatch: {directory}")
        if manifest.get("evaluation_sha256") != evaluation_sha256:
            raise ValueError(f"Evaluation checksum mismatch: {directory}")
        if manifest.get("model") != config["model"]:
            raise ValueError(f"Model configuration mismatch: {directory}")
        if manifests and manifest.get("adapter") != manifests[0].get("adapter"):
            raise ValueError(f"Adapter provenance mismatch: {directory}")
        if len(predictions) != int(manifest.get("requested_samples", -1)):
            raise ValueError(f"Shard prediction count is incomplete: {directory}")

        sharding = manifest.get("sharding", {})
        if sharding.get("strategy") != SHARDING_STRATEGY:
            raise ValueError(f"Unsupported sharding strategy: {directory}")

        for prediction in predictions:
            sample_id = str(prediction.get("sample_id"))
            if sample_id in predictions_by_id:
                raise ValueError(f"Duplicate sample_id across shards: {sample_id}")
            predictions_by_id[sample_id] = prediction

        manifests.append(manifest)
        shard_sources.append(
            {
                "directory": str(directory),
                "manifest_sha256": sha256_file(manifest_path),
                "predictions_sha256": sha256_file(predictions_path),
                "requested_samples": manifest["requested_samples"],
                "sharding": sharding,
            }
        )

    declared_counts = {int(item["sharding"]["num_shards"]) for item in manifests}
    shard_indices = {int(item["sharding"]["shard_index"]) for item in manifests}
    if declared_counts != {len(shard_dirs)}:
        raise ValueError("Shard manifests do not declare the supplied shard count")
    if shard_indices != set(range(len(shard_dirs))):
        raise ValueError("Shard indices must cover every value from zero to num_shards - 1")

    actual_ids = set(predictions_by_id)
    missing = sorted(expected_id_set - actual_ids)
    unexpected = sorted(actual_ids - expected_id_set)
    if missing or unexpected:
        raise ValueError(
            "Merged shards do not exactly cover the evaluation set; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    ordered_predictions = [predictions_by_id[sample_id] for sample_id in expected_ids]
    write_jsonl_atomic(destination / "predictions.jsonl", ordered_predictions)

    first = manifests[0]
    completed_at = datetime.now(UTC).isoformat()
    merged_manifest = {
        **first,
        "run_id": destination.name,
        "status": "completed",
        "started_at_utc": min(str(item["started_at_utc"]) for item in manifests),
        "completed_at_utc": completed_at,
        "requested_samples": len(expected_ids),
        "completed_samples": len(ordered_predictions),
        "sharding": {
            "strategy": SHARDING_STRATEGY,
            "num_shards": len(shard_dirs),
            "merged": True,
            "sources": shard_sources,
        },
        "hardware": {
            "merged_from_shards": [item.get("hardware", {}) for item in manifests]
        },
        "scheduler": {
            "merged_from_shards": [item.get("scheduler", {}) for item in manifests]
        },
    }
    write_json_atomic(destination / "manifest.json", merged_manifest)
    return destination
