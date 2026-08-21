from pathlib import Path

from afri_reasoning.io import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from afri_reasoning.merge import merge_baseline_shards
from afri_reasoning.run import SHARDING_STRATEGY, select_shard


def _samples() -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"p{problem}:{language}",
            "problem_id": f"p{problem}",
            "language": language,
        }
        for problem in range(4)
        for language in ("en", "fr", "sw", "wo")
    ]


def test_sharding_keeps_matched_languages_together() -> None:
    samples = _samples()
    first = select_shard(samples, num_shards=2, shard_index=0)
    second = select_shard(samples, num_shards=2, shard_index=1)

    assert len(first) == len(second) == 8
    first_problems = {sample["problem_id"] for sample in first}
    second_problems = {sample["problem_id"] for sample in second}
    assert first_problems.isdisjoint(second_problems)
    assert first_problems | second_problems == {"p0", "p1", "p2", "p3"}
    for problem_id in first_problems:
        assert {sample["language"] for sample in first if sample["problem_id"] == problem_id} == {
            "en",
            "fr",
            "sw",
            "wo",
        }


def test_merge_validates_and_restores_evaluation_order(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("model: test\n", encoding="utf-8")
    evaluation_path = tmp_path / "eval.jsonl"
    samples = _samples()
    write_jsonl_atomic(evaluation_path, samples)
    model = {"id": "test", "revision": "abc"}
    config = {
        "model": model,
        "paths": {"evaluation_file": "eval.jsonl"},
        "_meta": {"repo_root": str(tmp_path), "config_path": str(config_path)},
    }

    shard_directories = []
    for shard_index in range(2):
        shard_dir = tmp_path / f"shard-{shard_index}"
        shard_dir.mkdir()
        predictions = select_shard(samples, num_shards=2, shard_index=shard_index)
        write_jsonl_atomic(shard_dir / "predictions.jsonl", reversed(predictions))
        write_json_atomic(
            shard_dir / "manifest.json",
            {
                "status": "completed",
                "started_at_utc": f"2026-08-21T00:00:0{shard_index}+00:00",
                "configuration_sha256": sha256_file(config_path),
                "evaluation_sha256": sha256_file(evaluation_path),
                "model": model,
                "requested_samples": len(predictions),
                "sharding": {
                    "strategy": SHARDING_STRATEGY,
                    "num_shards": 2,
                    "shard_index": shard_index,
                },
                "hardware": {"gpu": shard_index},
                "scheduler": {"cuda_visible_devices": str(shard_index)},
            },
        )
        shard_directories.append(shard_dir)

    output = merge_baseline_shards(
        config, shard_dirs=shard_directories, output_dir=tmp_path / "merged"
    )

    assert [item["sample_id"] for item in read_jsonl(output / "predictions.jsonl")] == [
        item["sample_id"] for item in samples
    ]
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["completed_samples"] == 16
    assert manifest["sharding"]["merged"] is True
