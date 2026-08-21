from pathlib import Path

import pytest

from afri_reasoning.config import ConfigurationError
from afri_reasoning.io import write_json_atomic
from afri_reasoning.run import adapter_provenance
from afri_reasoning.training_config import load_training_config


def test_primary_training_config_keeps_multimodal_tower_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_training_config(root / "configs/training/unsloth_qlora_rslora.yaml")

    assert config["adapter"]["target_modules"] is None
    assert config["adapter"]["finetune_vision_layers"] is False
    assert config["adapter"]["finetune_language_layers"] is True


def test_all_linear_is_rejected_when_vision_is_declared_frozen(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs/training/unsloth_qlora_rslora.yaml").read_text(
        encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    config_path = tmp_path / "training.yaml"
    config_path.write_text(text.replace("target_modules: null", "target_modules: all-linear"))

    with pytest.raises(ConfigurationError, match="multimodal tower"):
        load_training_config(config_path)


def test_adapter_provenance_requires_and_hashes_completed_training(tmp_path: Path) -> None:
    run_dir = tmp_path / "training"
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text('{"r": 32}\n', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    write_json_atomic(
        run_dir / "training_manifest.json",
        {
            "status": "completed",
            "configuration_sha256": "config-sha",
            "git_commit": "commit",
            "adapter": {"method": "qlora", "use_rslora": True},
        },
    )

    result = adapter_provenance(adapter_dir)

    assert result["training_configuration_sha256"] == "config-sha"
    assert result["tree_sha256"]
    assert set(result["files"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
