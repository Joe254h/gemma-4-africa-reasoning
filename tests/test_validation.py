import json
from pathlib import Path

import pytest

from afri_reasoning.config import load_config
from afri_reasoning.prepare import _reference_answer
from afri_reasoning.validation import validate_evaluation


def _write_project(tmp_path: Path, mismatch: bool = False) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "eval").mkdir(parents=True)
    config_path = tmp_path / "configs" / "test.yaml"
    config_path.write_text(
        """
project: {name: test, milestone: test}
model: {id: test, revision: test, enable_thinking: true}
dataset:
  id: test
  revision: test
  split: test
  sample_size_per_language: 1
  languages:
    en: {dataset_config: eng, name: English}
    fr: {dataset_config: fra, name: French}
    sw: {dataset_config: swa, name: Swahili}
    wo: {dataset_config: wol, name: Wolof}
prompt: {system_template: test}
generation: {do_sample: false, max_new_tokens: 10, seed: 1}
paths:
  evaluation_file: data/eval/test.jsonl
  evaluation_manifest: data/eval/test.manifest.json
  artifacts_dir: artifacts
quality_gates:
  required_languages: [en, fr, sw, wo]
  require_matched_problem_ids: true
  require_equal_reference_answers: true
  require_thinking_content: true
  human_review_languages: [sw, wo]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    records = []
    for language in ["en", "fr", "sw", "wo"]:
        records.append(
            {
                "sample_id": f"p1:{language}",
                "problem_id": "p1",
                "language": language,
                "question": "q",
                "expected_answer": "8" if mismatch and language == "wo" else "7",
                "messages": [],
                "source": {"row_index": 1},
                "validation": {"project_native_review_status": "pending"},
            }
        )
    eval_path = tmp_path / "data" / "eval" / "test.jsonl"
    eval_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return config_path


def test_matched_four_language_evaluation_is_valid(tmp_path: Path) -> None:
    config = load_config(_write_project(tmp_path))
    result = validate_evaluation(config)
    assert result["status"] == "valid"
    assert result["records"] == 4


def test_reference_mismatch_fails_validation(tmp_path: Path) -> None:
    config = load_config(_write_project(tmp_path, mismatch=True))
    with pytest.raises(ValueError, match="inconsistent reference answers"):
        validate_evaluation(config)


def test_current_afrimgsm_answer_number_schema_is_supported() -> None:
    row = {"answer": "", "answer_number": "366", "question": "q"}
    assert _reference_answer(row) == "366"
