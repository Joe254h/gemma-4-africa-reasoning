from __future__ import annotations

import csv
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afri_reasoning.config import repo_path
from afri_reasoning.io import sha256_file, write_json_atomic, write_jsonl_atomic


def _reference_answer(row: dict[str, Any]) -> str:
    for field in ("answer_number", "answer"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Dataset row has no usable answer field; fields={sorted(row)}")


def _load_local_tsv(source_dir: Path, dataset_config: str, split: str) -> list[dict[str, Any]]:
    path = source_dir / "data" / dataset_config / f"{split}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Pinned AfriMGSM TSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _local_source_provenance(
    source_dir: Path, language_config: dict[str, Any], split: str, revision: str
) -> dict[str, Any]:
    git_head: str | None = None
    if (source_dir / ".git").exists():
        try:
            git_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=source_dir,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            git_head = None
        if git_head is not None and git_head != revision:
            raise ValueError(
                f"Local AfriMGSM checkout is at {git_head}, expected pinned revision {revision}"
            )

    files: dict[str, Any] = {}
    for language, details in language_config.items():
        path = source_dir / "data" / details["dataset_config"] / f"{split}.tsv"
        files[language] = {
            "path": path.relative_to(source_dir).as_posix(),
            "sha256": sha256_file(path),
        }
    return {"git_head": git_head, "files": files}


def prepare_evaluation(
    config: dict[str, Any],
    *,
    force: bool = False,
    source_dir: str | Path | None = None,
) -> dict[str, Any]:

    output_path = repo_path(config, config["paths"]["evaluation_file"])
    manifest_path = repo_path(config, config["paths"]["evaluation_manifest"])
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Evaluation file already exists: {output_path}. Use --force to regenerate it."
        )

    dataset_config = config["dataset"]
    language_config = dataset_config["languages"]
    loaded: dict[str, Any] = {}
    if source_dir is not None:
        local_source = Path(source_dir).expanduser().resolve()
        for language, details in language_config.items():
            loaded[language] = _load_local_tsv(
                local_source, details["dataset_config"], dataset_config["split"]
            )
        loading_method = "pinned-local-tsv"
        source_provenance = _local_source_provenance(
            local_source,
            language_config,
            dataset_config["split"],
            dataset_config["revision"],
        )
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Dataset preparation requires the baseline dependencies. Install "
                "requirements/hpc.txt, or pass --source-dir to a pinned AfriMGSM checkout."
            ) from exc
        for language, details in language_config.items():
            loaded[language] = load_dataset(
                dataset_config["id"],
                details["dataset_config"],
                split=dataset_config["split"],
                revision=dataset_config["revision"],
            )
        loading_method = "huggingface-datasets"
        source_provenance = {
            "dataset_fingerprints": {
                language: getattr(dataset, "_fingerprint", None)
                for language, dataset in loaded.items()
            }
        }

    lengths = {language: len(dataset) for language, dataset in loaded.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Language split lengths are not equal: {lengths}")

    population_size = next(iter(lengths.values()))
    sample_size = int(dataset_config["sample_size_per_language"])
    if sample_size > population_size:
        raise ValueError(f"Requested {sample_size} rows from a split of {population_size}")

    randomizer = random.Random(int(dataset_config["sample_seed"]))
    selected_indices = sorted(randomizer.sample(range(population_size), sample_size))
    review_languages = set(config["quality_gates"]["human_review_languages"])

    records: list[dict[str, Any]] = []
    for source_index in selected_indices:
        answers = {
            language: _reference_answer(dict(loaded[language][source_index]))
            for language in loaded
        }
        if len(set(answers.values())) != 1:
            raise ValueError(
                f"Reference-answer mismatch at source row {source_index}: {answers}"
            )

        problem_id = f"afrimgsm-{dataset_config['split']}-{source_index:04d}"
        for language, details in language_config.items():
            row = loaded[language][source_index]
            language_name = details["name"]
            system_prompt = config["prompt"]["system_template"].format(
                language_name=language_name
            )
            records.append(
                {
                    "schema_version": "1.0",
                    "sample_id": f"{problem_id}:{language}",
                    "problem_id": problem_id,
                    "language": language,
                    "language_name": language_name,
                    "question": str(row["question"]).strip(),
                    "expected_answer": answers[language],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(row["question"]).strip()},
                    ],
                    "source": {
                        "dataset_id": dataset_config["id"],
                        "dataset_revision": dataset_config["revision"],
                        "dataset_config": details["dataset_config"],
                        "split": dataset_config["split"],
                        "row_index": source_index,
                        "license": "apache-2.0",
                    },
                    "validation": {
                        "source_benchmark_translation": True,
                        "project_native_review_required": language in review_languages,
                        "project_native_review_status": (
                            "pending" if language in review_languages else "not-required"
                        ),
                        "reviewer": None,
                        "review_date": None,
                        "notes": None,
                    },
                }
            )

    write_jsonl_atomic(output_path, records)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": dataset_config["id"],
        "dataset_revision": dataset_config["revision"],
        "loading_method": loading_method,
        "split": dataset_config["split"],
        "sample_seed": dataset_config["sample_seed"],
        "selected_source_indices": selected_indices,
        "languages": list(language_config),
        "records": len(records),
        "problems": sample_size,
        "evaluation_file": output_path.relative_to(
            Path(config["_meta"]["repo_root"])
        ).as_posix(),
        "evaluation_sha256": sha256_file(output_path),
        "source_provenance": source_provenance,
        "native_review_policy": {
            "languages_requiring_project_review": sorted(review_languages),
            "status": "pending",
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
