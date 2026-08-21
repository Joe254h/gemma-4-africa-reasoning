from pathlib import Path

import pytest

from afri_reasoning.io import write_jsonl_atomic
from afri_reasoning.training_data import (
    TrainingDataError,
    build_supervised_features,
    validate_training_corpus,
)

LANGUAGES = ("en", "fr", "sw", "wo")


def _record(
    group: str,
    language: str,
    split: str,
    *,
    verified: bool = True,
    user_text: str | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "sample_id": f"{group}:{language}:{split}",
        "source_group_id": group,
        "language": language,
        "split": split,
        "messages": [
            {"role": "system", "content": "Reason carefully."},
            {"role": "user", "content": user_text or f"Question {group} in {language}?"},
            {
                "role": "assistant",
                "reasoning": f"Reasoning for {group} in {language}.",
                "content": f"Answer for {group} in {language}.",
            },
        ],
        "source": {
            "dataset_id": "source",
            "revision": "abc",
            "split": "train",
            "source_id": group,
            "license": "apache-2.0",
        },
        "validation": {
            "training_ready": True,
            "format_valid": True,
            "semantic_accuracy": True,
            "reasoning_consistent": True,
            "language_verified": True,
            "human_verified": verified,
            "verifier": "reviewer" if verified else None,
            "verification_date": "2026-08-21" if verified else None,
        },
    }


def _project(tmp_path: Path) -> tuple[dict, Path, Path]:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    write_jsonl_atomic(
        evaluation_path,
        [
            {
                "sample_id": "eval:en",
                "problem_id": "eval-problem",
                "language": "en",
                "question": "A held-out evaluation question?",
            }
        ],
    )
    config = {
        "dataset": {"schema_version": "1.0"},
        "paths": {
            "train_file": str(train_path),
            "validation_file": str(validation_path),
            "evaluation_file": str(evaluation_path),
        },
        "quality_gates": {
            "required_languages": list(LANGUAGES),
            "human_review_languages": ["sw", "wo"],
            "require_parallel_groups": True,
            "minimum_records_per_language": 1,
            "maximum_language_imbalance_ratio": 1.0,
        },
        "_meta": {"repo_root": str(tmp_path)},
    }
    return config, train_path, validation_path


def test_valid_parallel_corpus_passes_all_gates(tmp_path: Path) -> None:
    config, train_path, validation_path = _project(tmp_path)
    write_jsonl_atomic(train_path, [_record("train-group", lang, "train") for lang in LANGUAGES])
    write_jsonl_atomic(
        validation_path,
        [_record("validation-group", lang, "validation") for lang in LANGUAGES],
    )

    result = validate_training_corpus(config)

    assert result["status"] == "valid"
    assert result["counts"]["train"] == {lang: 1 for lang in LANGUAGES}
    assert result["source_groups"] == {"train": 1, "validation": 1}


def test_reported_protocol_rejects_wrong_corpus_size(tmp_path: Path) -> None:
    config, train_path, validation_path = _project(tmp_path)
    config["dataset"].update(
        {"target_train_records": 8, "target_validation_fraction": 0.5}
    )
    config["quality_gates"]["require_target_size"] = True
    write_jsonl_atomic(train_path, [_record("train-group", lang, "train") for lang in LANGUAGES])
    write_jsonl_atomic(
        validation_path,
        [_record("validation-group", lang, "validation") for lang in LANGUAGES],
    )

    with pytest.raises(TrainingDataError, match="Corpus size"):
        validate_training_corpus(config)


def test_unverified_wolof_is_rejected(tmp_path: Path) -> None:
    config, train_path, validation_path = _project(tmp_path)
    train = [_record("train-group", lang, "train") for lang in LANGUAGES]
    train[-1] = _record("train-group", "wo", "train", verified=False)
    write_jsonl_atomic(train_path, train)
    write_jsonl_atomic(
        validation_path,
        [_record("validation-group", lang, "validation") for lang in LANGUAGES],
    )

    with pytest.raises(TrainingDataError, match="native human verification"):
        validate_training_corpus(config)


def test_parallel_group_cannot_cross_splits(tmp_path: Path) -> None:
    config, train_path, validation_path = _project(tmp_path)
    write_jsonl_atomic(train_path, [_record("shared", lang, "train") for lang in LANGUAGES])
    write_jsonl_atomic(
        validation_path,
        [_record("shared", lang, "validation") for lang in LANGUAGES],
    )

    with pytest.raises(TrainingDataError, match="cross train/validation"):
        validate_training_corpus(config)


def test_parallel_group_requires_exactly_one_record_per_language(tmp_path: Path) -> None:
    config, train_path, validation_path = _project(tmp_path)
    train = [_record("train-group", lang, "train") for lang in LANGUAGES]
    duplicate = _record("train-group", "en", "train")
    duplicate["sample_id"] = "train-group:en:duplicate"
    train.append(duplicate)
    write_jsonl_atomic(train_path, train)
    write_jsonl_atomic(
        validation_path,
        [_record("validation-group", lang, "validation") for lang in LANGUAGES],
    )

    with pytest.raises(TrainingDataError, match="expected"):
        validate_training_corpus(config)


def test_parallel_group_cannot_mix_source_identities(tmp_path: Path) -> None:
    config, train_path, validation_path = _project(tmp_path)
    train = [_record("train-group", lang, "train") for lang in LANGUAGES]
    train[1]["source"]["source_id"] = "different-source"
    write_jsonl_atomic(train_path, train)
    write_jsonl_atomic(
        validation_path,
        [_record("validation-group", lang, "validation") for lang in LANGUAGES],
    )

    with pytest.raises(TrainingDataError, match="mixes source identities"):
        validate_training_corpus(config)


def test_exact_evaluation_prompt_leakage_is_rejected(tmp_path: Path) -> None:
    config, train_path, validation_path = _project(tmp_path)
    train = [_record("train-group", lang, "train") for lang in LANGUAGES]
    train[0] = _record(
        "train-group",
        "en",
        "train",
        user_text="A held-out evaluation question?",
    )
    write_jsonl_atomic(train_path, train)
    write_jsonl_atomic(
        validation_path,
        [_record("validation-group", lang, "validation") for lang in LANGUAGES],
    )

    with pytest.raises(TrainingDataError, match="evaluation prompt leakage"):
        validate_training_corpus(config)


class _FakeProcessor:
    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> dict:
        if kwargs["add_generation_prompt"]:
            return {"input_ids": [[1, 2, 3]]}
        return {"input_ids": [[1, 2, 3, 4, 5]]}


def test_reasoning_features_mask_prompt_and_keep_assistant_loss() -> None:
    features = build_supervised_features(
        _record("group", "sw", "train"),
        _FakeProcessor(),
        max_seq_length=8,
    )

    assert features["input_ids"] == [1, 2, 3, 4, 5]
    assert features["labels"] == [-100, -100, -100, 4, 5]
