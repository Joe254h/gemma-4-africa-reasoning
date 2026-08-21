from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("The configuration root must be a mapping.")

    required_sections = {
        "project",
        "model",
        "dataset",
        "prompt",
        "generation",
        "paths",
        "quality_gates",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {', '.join(missing)}")

    languages = config["dataset"].get("languages", {})
    required_languages = set(config["quality_gates"].get("required_languages", []))
    if set(languages) != required_languages:
        raise ConfigurationError(
            "dataset.languages must exactly match quality_gates.required_languages"
        )

    config["_meta"] = {
        "config_path": str(config_path),
        "repo_root": str(config_path.parent.parent),
    }
    return config


def repo_path(config: dict[str, Any], configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_meta"]["repo_root"]) / path
