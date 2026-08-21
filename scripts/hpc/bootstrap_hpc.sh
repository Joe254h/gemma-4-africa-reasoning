#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_PATH="${VENV_PATH:-/scratch/${USER}/envs/gemma4-reasoning}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Load the cluster Python module, then rerun." >&2
  exit 2
fi

echo "Project: ${PROJECT_ROOT}"
echo "Environment: ${VENV_PATH}"
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  echo "Installing PyTorch from ${TORCH_INDEX_URL}"
  python -m pip install torch --index-url "${TORCH_INDEX_URL}"
else
  echo "Installing the default PyPI PyTorch build."
  echo "Set TORCH_INDEX_URL first if the HPC requires a specific CUDA wheel."
  python -m pip install torch
fi

python -m pip install -r "${PROJECT_ROOT}/requirements/hpc.txt"
python -m pip install -e "${PROJECT_ROOT}" --no-deps

python - <<'PY'
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"torch CUDA build={torch.version.cuda}")
print(f"CUDA currently available={torch.cuda.is_available()}")
print("CUDA may be false on the login node; verify again inside the Slurm GPU job.")
PY

echo "Environment ready. Authenticate with: hf auth login"
