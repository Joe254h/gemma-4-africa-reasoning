from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afri_reasoning.config import repo_path
from afri_reasoning.inference import GemmaThinkingRunner
from afri_reasoning.io import (
    append_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from afri_reasoning.provenance import build_run_manifest

SHARDING_STRATEGY = "matched-problem-round-robin-v1"


def adapter_provenance(adapter_dir: str | Path) -> dict[str, Any]:
    """Validate a completed adapter and fingerprint every persisted adapter file."""
    adapter_path = Path(adapter_dir).expanduser().resolve()
    training_manifest_path = adapter_path.parent / "training_manifest.json"
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
    if not training_manifest_path.is_file():
        raise FileNotFoundError(
            f"Training manifest not found beside adapter: {training_manifest_path}"
        )
    training_manifest = read_json(training_manifest_path)
    if training_manifest.get("status") != "completed":
        raise RuntimeError("Adapter evaluation requires a completed training manifest")

    files = sorted(path for path in adapter_path.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"Adapter directory is empty: {adapter_path}")
    file_hashes = {
        path.relative_to(adapter_path).as_posix(): sha256_file(path) for path in files
    }
    tree_digest = hashlib.sha256(
        "\n".join(f"{name}\t{digest}" for name, digest in file_hashes.items()).encode("utf-8")
    ).hexdigest()
    return {
        "directory": str(adapter_path),
        "tree_sha256": tree_digest,
        "files": file_hashes,
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "training_configuration_sha256": training_manifest.get("configuration_sha256"),
        "training_git_commit": training_manifest.get("git_commit"),
        "method": training_manifest.get("adapter"),
    }


def select_shard(
    samples: list[dict[str, Any]], *, num_shards: int, shard_index: int
) -> list[dict[str, Any]]:
    """Select a deterministic shard while keeping language variants together."""
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be between 0 and num_shards - 1")

    problem_shards: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for sample in samples:
        problem_id = str(sample["problem_id"])
        if problem_id not in problem_shards:
            problem_shards[problem_id] = len(problem_shards) % num_shards
        if problem_shards[problem_id] == shard_index:
            selected.append(sample)
    return selected


def run_baseline(
    config: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
    limit_per_language: int | None = None,
    local_files_only: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
    adapter_dir: str | Path | None = None,
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

    samples = select_shard(samples, num_shards=num_shards, shard_index=shard_index)
    if not samples:
        raise ValueError(
            f"Shard {shard_index} contains no samples; reduce num_shards or remove the limit"
        )

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
    manifest["evaluation_stage"] = "adapted" if adapter_dir is not None else "zero-shot"
    manifest["adapter"] = adapter_provenance(adapter_dir) if adapter_dir is not None else None
    manifest["requested_samples"] = len(samples)
    manifest["limit_per_language"] = limit_per_language
    manifest["sharding"] = {
        "strategy": SHARDING_STRATEGY,
        "num_shards": num_shards,
        "shard_index": shard_index,
    }

    if manifest_path.exists():
        previous = read_json(manifest_path)
        invariant_fields = ["evaluation_sha256", "configuration_sha256", "adapter"]
        mismatches = [
            field for field in invariant_fields if previous.get(field) != manifest.get(field)
        ]
        if previous.get("model") != manifest.get("model"):
            mismatches.append("model")
        if previous.get("sharding") != manifest.get("sharding"):
            mismatches.append("sharding")
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
        runner = GemmaThinkingRunner(
            config,
            local_files_only=local_files_only,
            adapter_dir=adapter_dir,
        )
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
