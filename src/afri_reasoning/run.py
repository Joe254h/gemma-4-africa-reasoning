from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afri_reasoning.config import repo_path
from afri_reasoning.inference import GemmaThinkingRunner
from afri_reasoning.io import append_jsonl, read_json, read_jsonl, write_json_atomic
from afri_reasoning.provenance import build_run_manifest


def run_baseline(
    config: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
    limit_per_language: int | None = None,
    local_files_only: bool = False,
) -> Path:
    evaluation_path = repo_path(config, config["paths"]["evaluation_file"])
    samples = read_jsonl(evaluation_path)
    if limit_per_language is not None:
        if limit_per_language < 1:
            raise ValueError("limit_per_language must be at least 1")
        counts: dict[str, int] = defaultdict(int)
        selected = []
        for sample in samples:
            language = sample["language"]
            if counts[language] < limit_per_language:
                selected.append(sample)
                counts[language] += 1
        samples = selected

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if run_dir is None:
        output_root = repo_path(config, config["paths"]["artifacts_dir"])
        output_dir = output_root / run_id
    else:
        output_dir = Path(run_dir).expanduser().resolve()
        run_id = output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    predictions_path = output_dir / "predictions.jsonl"
    manifest = build_run_manifest(config, run_id, evaluation_path)
    manifest["requested_samples"] = len(samples)
    manifest["limit_per_language"] = limit_per_language

    if manifest_path.exists():
        previous = read_json(manifest_path)
        invariant_fields = ["evaluation_sha256", "configuration_sha256"]
        mismatches = [
            field for field in invariant_fields if previous.get(field) != manifest.get(field)
        ]
        if previous.get("model") != manifest.get("model"):
            mismatches.append("model")
        if mismatches:
            raise RuntimeError(
                "Refusing to resume into a run directory from a different experiment; "
                f"mismatched fields: {sorted(mismatches)}"
            )
        manifest["resumed_from_status"] = previous.get("status")
        manifest["original_started_at_utc"] = previous.get("started_at_utc")
    write_json_atomic(manifest_path, manifest)

    completed_ids: set[str] = set()
    if predictions_path.exists():
        completed_ids = {record["sample_id"] for record in read_jsonl(predictions_path)}

    try:
        runner = GemmaThinkingRunner(config, local_files_only=local_files_only)
        total = len(samples)
        for position, sample in enumerate(samples, start=1):
            if sample["sample_id"] in completed_ids:
                print(f"[{position}/{total}] skip completed {sample['sample_id']}", flush=True)
                continue
            print(f"[{position}/{total}] generate {sample['sample_id']}", flush=True)
            prediction = runner.generate(sample)
            prediction["run_id"] = run_id
            append_jsonl(predictions_path, prediction)
            print(
                f"[{position}/{total}] saved {sample['sample_id']} "
                f"({prediction['latency_seconds']:.2f}s, parse={prediction['parse_valid']})",
                flush=True,
            )
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        manifest["completed_samples"] = len(read_jsonl(predictions_path))
        write_json_atomic(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(UTC).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(manifest_path, manifest)
        raise

    return output_dir
