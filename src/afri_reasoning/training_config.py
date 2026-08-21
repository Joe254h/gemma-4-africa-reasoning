from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from afri_reasoning.config import ConfigurationError


def _find_repo_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigurationError(f"Could not locate repository root above {config_path}")


def load_training_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Training configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("The training configuration root must be a mapping")

    required_sections = {
        "project",
        "model",
        "dataset",
        "adapter",
        "training",
        "runtime",
        "paths",
        "quality_gates",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ConfigurationError(f"Missing training sections: {', '.join(missing)}")

    required_languages = set(config["quality_gates"].get("required_languages", []))
    configured_languages = set(config["dataset"].get("languages", []))
    if configured_languages != required_languages:
        raise ConfigurationError(
            "dataset.languages must exactly match quality_gates.required_languages"
        )
    language_count = len(required_languages)
    if language_count == 0:
        raise ConfigurationError("At least one required language must be configured")
    target_train = int(config["dataset"]["target_train_records"])
    target_validation = round(
        target_train * float(config["dataset"]["target_validation_fraction"])
    )
    if target_train % language_count or target_validation % language_count:
        raise ConfigurationError(
            "Target train and validation sizes must divide evenly across languages"
        )
    if config["adapter"].get("method") != "qlora":
        raise ConfigurationError("The primary training protocol must use method=qlora")
    if not config["model"].get("load_in_4bit"):
        raise ConfigurationError("QLoRA requires model.load_in_4bit=true")

    adapter = config["adapter"]
    if (
        adapter.get("target_modules") == "all-linear"
        and not adapter.get("finetune_vision_layers")
    ):
        raise ConfigurationError(
            "Unsloth target_modules=all-linear also enables the multimodal tower; "
            "use target_modules=null for text-only adapter selection"
        )

    config["_meta"] = {
        "config_path": str(config_path),
        "repo_root": str(_find_repo_root(config_path)),
    }
    return config


def training_path(config: dict[str, Any], configured_path: str | Path) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_meta"]["repo_root"]) / path).resolve()
