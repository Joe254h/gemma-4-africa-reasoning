#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"

python -m pip install --no-cache-dir -r requirements/kaggle-unsloth.txt
python -m pip install -e . --no-deps
python -m pip check

python - <<'PY'
import importlib.metadata
import torch

packages = (
    "unsloth",
    "unsloth_zoo",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
)
print(f"torch={torch.__version__}; cuda={torch.version.cuda}")
for package in packages:
    print(f"{package}={importlib.metadata.version(package)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable after installation")
print(f"visible_gpus={torch.cuda.device_count()}")
PY
