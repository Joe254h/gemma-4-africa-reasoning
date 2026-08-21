from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from afri_reasoning.io import read_jsonl, sha256_file
from afri_reasoning.training_config import training_path

FORBIDDEN_FORMAT_TOKENS = (
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|think|>",
)


class TrainingDataError(ValueError):
    """Raised when curated training data violates the frozen data contract."""


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def text_fingerprint(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def _message(records: Sequence[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [message for message in records if message.get("role") == role]
    if len(matches) != 1:
        raise TrainingDataError(f"Expected exactly one {role} message, found {len(matches)}")
    return matches[0]


def _require_clean_text(sample_id: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingDataError(f"{sample_id}: {field} must be non-empty text")
    forbidden = [token for token in FORBIDDEN_FORMAT_TOKENS if token in value]
    if forbidden:
        raise TrainingDataError(
            f"{sample_id}: {field} contains reserved Gemma tokens: {forbidden}"
        )
    return value.strip()


def _validate_record(
    record: dict[str, Any],
    *,
    expected_split: str,
    expected_schema_version: str,
    expected_dataset_id: str | None,
    expected_dataset_revision: str | None,
    human_review_languages: set[str],
) -> tuple[str, str, str, tuple[str, str, str]]:
    required = {
        "schema_version",
        "sample_id",
        "source_group_id",
        "language",
        "split",
        "messages",
        "source",
        "validation",
    }
    missing = sorted(required - set(record))
    sample_id = str(record.get("sample_id", "<unknown>"))
    if missing:
        raise TrainingDataError(f"{sample_id}: missing fields {missing}")
    if record["schema_version"] != expected_schema_version:
        raise TrainingDataError(
            f"{sample_id}: schema_version={record['schema_version']!r}, "
            f"expected {expected_schema_version!r}"
        )
    if not isinstance(record["sample_id"], str) or not record["sample_id"].strip():
        raise TrainingDataError("sample_id must be non-empty text")
    if not isinstance(record["source_group_id"], str) or not record[
        "source_group_id"
    ].strip():
        raise TrainingDataError(f"{sample_id}: source_group_id must be non-empty text")
    if record["split"] != expected_split:
        raise TrainingDataError(
            f"{sample_id}: split={record['split']!r}, expected {expected_split!r}"
        )

    messages = record["messages"]
    if not isinstance(messages, list):
        raise TrainingDataError(f"{sample_id}: messages must be a list")
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise TrainingDataError(
            f"{sample_id}: message roles must be exactly system, user, assistant"
        )

    system = _message(messages, "system")
    user = _message(messages, "user")
    assistant = _message(messages, "assistant")
    _require_clean_text(sample_id, "system.content", system.get("content"))
    user_text = _require_clean_text(sample_id, "user.content", user.get("content"))
    _require_clean_text(sample_id, "assistant.reasoning", assistant.get("reasoning"))
    _require_clean_text(sample_id, "assistant.content", assistant.get("content"))
    if "thinking" in assistant:
        raise TrainingDataError(
            f"{sample_id}: use assistant.reasoning, not assistant.thinking, for Gemma 4"
        )

    source = record["source"]
    source_required = {"dataset_id", "revision", "split", "source_id", "license"}
    if not isinstance(source, dict) or source_required - set(source):
        raise TrainingDataError(f"{sample_id}: incomplete source provenance")
    if any(
        not isinstance(source[field], str) or not source[field].strip()
        for field in source_required
    ):
        raise TrainingDataError(f"{sample_id}: source provenance values must be non-empty text")
    if expected_dataset_id is not None and source["dataset_id"] != expected_dataset_id:
        raise TrainingDataError(
            f"{sample_id}: source dataset_id={source['dataset_id']!r}, "
            f"expected {expected_dataset_id!r}"
        )
    if (
        expected_dataset_revision is not None
        and source["revision"] != expected_dataset_revision
    ):
        raise TrainingDataError(
            f"{sample_id}: source revision={source['revision']!r}, "
            f"expected {expected_dataset_revision!r}"
        )

    validation = record["validation"]
    validation_flags = (
        "training_ready",
        "format_valid",
        "semantic_accuracy",
        "reasoning_consistent",
        "language_verified",
    )
    if not isinstance(validation, dict) or any(
        validation.get(flag) is not True for flag in validation_flags
    ):
        raise TrainingDataError(f"{sample_id}: all validation gates must be true")

    language = str(record["language"])
    if language in human_review_languages:
        if validation.get("human_verified") is not True:
            raise TrainingDataError(f"{sample_id}: native human verification is required")
        if not validation.get("verifier") or not validation.get("verification_date"):
            raise TrainingDataError(f"{sample_id}: verifier and verification_date are required")
        try:
            date.fromisoformat(str(validation["verification_date"]))
        except ValueError as exc:
            raise TrainingDataError(
                f"{sample_id}: verification_date must use YYYY-MM-DD"
            ) from exc

    source_signature = tuple(
        str(source[field]) for field in ("dataset_id", "revision", "source_id")
    )
    return (
        sample_id,
        str(record["source_group_id"]),
        text_fingerprint(user_text),
        source_signature,
    )


def validate_training_corpus(
    config: dict[str, Any],
    *,
    train_file: str | Path | None = None,
    validation_file: str | Path | None = None,
) -> dict[str, Any]:
    train_path = training_path(config, train_file or config["paths"]["train_file"])
    validation_path = training_path(
        config, validation_file or config["paths"]["validation_file"]
    )
    evaluation_path = training_path(config, config["paths"]["evaluation_file"])
    train_records = read_jsonl(train_path)
    validation_records = read_jsonl(validation_path)
    evaluation_records = read_jsonl(evaluation_path)
    required_languages = set(config["quality_gates"]["required_languages"])
    human_review_languages = set(config["quality_gates"]["human_review_languages"])

    all_ids: set[str] = set()
    seen_prompts: dict[tuple[str, str], tuple[str, str]] = {}
    groups_by_split: dict[str, dict[str, Counter[str]]] = {
        "train": defaultdict(Counter),
        "validation": defaultdict(Counter),
    }
    source_by_group: dict[str, tuple[str, str, str]] = {}
    counts_by_split: dict[str, Counter[str]] = {}

    for split, records in (("train", train_records), ("validation", validation_records)):
        counts: Counter[str] = Counter()
        for record in records:
            language = str(record.get("language"))
            if language not in required_languages:
                raise TrainingDataError(
                    f"{record.get('sample_id')}: unsupported language {language!r}"
                )
            sample_id, group_id, prompt_hash, source_signature = _validate_record(
                record,
                expected_split=split,
                expected_schema_version=str(config["dataset"]["schema_version"]),
                expected_dataset_id=config["dataset"].get("id"),
                expected_dataset_revision=config["dataset"].get("revision"),
                human_review_languages=human_review_languages,
            )
            if sample_id in all_ids:
                raise TrainingDataError(f"Duplicate sample_id across splits: {sample_id}")
            all_ids.add(sample_id)
            prompt_key = (language, prompt_hash)
            previous = seen_prompts.get(prompt_key)
            if previous is not None and previous[1] != group_id:
                raise TrainingDataError(
                    f"Duplicate normalized prompt: {sample_id} and {previous[0]}"
                )
            seen_prompts[prompt_key] = (sample_id, group_id)
            previous_source = source_by_group.get(group_id)
            if previous_source is not None and previous_source != source_signature:
                raise TrainingDataError(
                    f"Parallel group {group_id} mixes source identities: "
                    f"{previous_source} and {source_signature}"
                )
            source_by_group[group_id] = source_signature
            groups_by_split[split][group_id][language] += 1
            counts[language] += 1
        counts_by_split[split] = counts

    if set(groups_by_split["train"]) & set(groups_by_split["validation"]):
        overlap = sorted(set(groups_by_split["train"]) & set(groups_by_split["validation"]))
        raise TrainingDataError(f"Source groups cross train/validation splits: {overlap[:10]}")

    if config["quality_gates"].get("require_parallel_groups", True):
        for split, groups in groups_by_split.items():
            expected_counts = Counter({language: 1 for language in required_languages})
            for group_id, language_counts in groups.items():
                if language_counts != expected_counts:
                    raise TrainingDataError(
                        f"{split} group {group_id} has {dict(language_counts)}, "
                        f"expected {dict(expected_counts)}"
                    )

    minimum = int(config["quality_gates"].get("minimum_records_per_language", 1))
    maximum_ratio = float(config["quality_gates"].get("maximum_language_imbalance_ratio", 1.0))
    for split, counts in counts_by_split.items():
        values = [counts[language] for language in sorted(required_languages)]
        if min(values, default=0) < minimum:
            raise TrainingDataError(f"{split} has fewer than {minimum} records per language")
        if min(values) == 0 or max(values) / min(values) > maximum_ratio:
            raise TrainingDataError(f"{split} language counts are imbalanced: {dict(counts)}")

    if config["quality_gates"].get("require_target_size", False):
        target_train = int(config["dataset"]["target_train_records"])
        target_validation = round(
            target_train * float(config["dataset"]["target_validation_fraction"])
        )
        actual_sizes = (len(train_records), len(validation_records))
        expected_sizes = (target_train, target_validation)
        if actual_sizes != expected_sizes:
            raise TrainingDataError(
                "Corpus size does not match the reported protocol; "
                f"train/validation={actual_sizes}, expected={expected_sizes}"
            )

    evaluation_problem_ids = {str(record["problem_id"]) for record in evaluation_records}
    evaluation_prompt_hashes = {
        text_fingerprint(str(record["question"])) for record in evaluation_records
    }
    for record in (*train_records, *validation_records):
        sample_id = str(record["sample_id"])
        if str(record.get("problem_id", "")) in evaluation_problem_ids:
            raise TrainingDataError(f"{sample_id}: evaluation problem_id leakage")
        user_text = _message(record["messages"], "user")["content"]
        if text_fingerprint(user_text) in evaluation_prompt_hashes:
            raise TrainingDataError(f"{sample_id}: exact normalized evaluation prompt leakage")

    corpus_digest = hashlib.sha256(
        (sha256_file(train_path) + sha256_file(validation_path)).encode("ascii")
    ).hexdigest()
    return {
        "status": "valid",
        "schema_version": config["dataset"]["schema_version"],
        "train_file": str(train_path),
        "validation_file": str(validation_path),
        "evaluation_file": str(evaluation_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "corpus_sha256": corpus_digest,
        "counts": {
            split: dict(sorted(counts.items())) for split, counts in counts_by_split.items()
        },
        "source_groups": {
            split: len(groups) for split, groups in groups_by_split.items()
        },
    }


def _token_ids(rendered: Any) -> list[int]:
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if isinstance(rendered, Sequence) and rendered and isinstance(rendered[0], Sequence):
        if len(rendered) != 1:
            raise TrainingDataError("Expected a single tokenized conversation")
        rendered = rendered[0]
    if not isinstance(rendered, Sequence):
        raise TrainingDataError("Processor did not return token IDs")
    return [int(token_id) for token_id in rendered]


def build_supervised_features(
    record: dict[str, Any], processor: Any, *, max_seq_length: int
) -> dict[str, list[int]]:
    """Render native Gemma reasoning and mask all prompt tokens from the loss."""
    prompt_messages = record["messages"][:-1]
    prompt_ids = _token_ids(
        processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )
    full_ids = _token_ids(
        processor.apply_chat_template(
            record["messages"],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise TrainingDataError(
            f"{record['sample_id']}: full conversation does not preserve the prompt prefix"
        )
    if len(full_ids) > max_seq_length:
        raise TrainingDataError(
            f"{record['sample_id']}: {len(full_ids)} tokens exceed max_seq_length={max_seq_length}"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise TrainingDataError(f"{record['sample_id']}: no assistant tokens remain for loss")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
