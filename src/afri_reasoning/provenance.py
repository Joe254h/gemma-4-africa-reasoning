from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afri_reasoning.io import sha256_file


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str | None]:
    packages = ["torch", "transformers", "accelerate", "datasets", "huggingface-hub", "PyYAML"]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_run_manifest(
    config: dict[str, Any], run_id: str, evaluation_path: Path
) -> dict[str, Any]:
    repo_root = Path(config["_meta"]["repo_root"])
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "project": config["project"],
        "model": config["model"],
        "dataset": {
            "id": config["dataset"]["id"],
            "revision": config["dataset"]["revision"],
            "split": config["dataset"]["split"],
        },
        "generation": config["generation"],
        "quality_gates": config["quality_gates"],
        "configuration_file": config["_meta"]["config_path"],
        "configuration_sha256": sha256_file(config["_meta"]["config_path"]),
        "evaluation_file": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "git_commit": _git_commit(repo_root),
        "packages": _package_versions(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "machine": platform.machine(),
        },
        "scheduler": {
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
            "slurm_node_list": os.getenv("SLURM_JOB_NODELIST"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
    }

    try:
        import torch

        manifest["hardware"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        manifest["hardware"] = {"cuda_available": False, "error": "torch not installed"}

    return manifest
