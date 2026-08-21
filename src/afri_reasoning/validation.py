from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from afri_reasoning.config import repo_path
from afri_reasoning.io import read_jsonl, sha256_file


def validate_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    path = repo_path(config, config["paths"]["evaluation_file"])
    records = read_jsonl(path)
    required = set(config["quality_gates"]["required_languages"])
    sample_ids = [record.get("sample_id") for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [item for item, count in Counter(sample_ids).items() if count > 1]
        raise ValueError(f"Duplicate sample IDs: {duplicates[:10]}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        missing = {
            "sample_id",
            "problem_id",
            "language",
            "question",
            "expected_answer",
            "messages",
            "source",
            "validation",
        } - set(record)
        if missing:
            raise ValueError(f"{record.get('sample_id')} is missing fields: {sorted(missing)}")
        grouped[record["problem_id"]].append(record)

    for problem_id, variants in grouped.items():
        languages = {variant["language"] for variant in variants}
        if languages != required:
            raise ValueError(
                f"{problem_id} has languages {sorted(languages)}, expected {sorted(required)}"
            )
        answers = {str(variant["expected_answer"]).strip() for variant in variants}
        if len(answers) != 1:
            raise ValueError(f"{problem_id} has inconsistent reference answers: {answers}")
        source_indices = {variant["source"]["row_index"] for variant in variants}
        if len(source_indices) != 1:
            raise ValueError(f"{problem_id} is not matched to one source row")

    expected_per_language = int(config["dataset"]["sample_size_per_language"])
    counts = Counter(record["language"] for record in records)
    if any(counts[language] != expected_per_language for language in required):
        raise ValueError(
            f"Unexpected per-language counts: {dict(counts)}; expected {expected_per_language}"
        )

    return {
        "status": "valid",
        "evaluation_file": str(path),
        "sha256": sha256_file(path),
        "records": len(records),
        "problems": len(grouped),
        "counts_by_language": dict(sorted(counts.items())),
        "pending_project_native_review": sum(
            record["validation"]["project_native_review_status"] == "pending"
            for record in records
        ),
    }
