#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pipeline-test}"
if [[ "${MODE}" != "pipeline-test" && "${MODE}" != "full" ]]; then
  echo "Usage: bash scripts/kaggle/run_training_t4x2.sh [pipeline-test|full]" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-configs/training/unsloth_qlora_rslora.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/gemma4-runs}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${OUTPUT_ROOT}/baseline-v1/merged}"

discover_unique_file() {
  local filename="$1"
  local -a matches=()
  mapfile -t matches < <(find /kaggle/input -type f -name "${filename}" 2>/dev/null | sort)
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "Expected exactly one ${filename} under /kaggle/input; found ${#matches[@]}" >&2
    printf '%s\n' "${matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

TRAIN_FILE="${TRAIN_FILE:-$(discover_unique_file train.jsonl)}"
VALIDATION_FILE="${VALIDATION_FILE:-$(discover_unique_file validation.jsonl)}"

cd "${PROJECT_ROOT}"
export HF_HOME="${HF_HOME:-/kaggle/temp/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_ROOT}" "${HF_HOME}"

python - <<'PY'
import re
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"Kaggle T4 x2 is required; detected {torch.cuda.device_count()} GPU(s)")
for index in range(2):
    properties = torch.cuda.get_device_properties(index)
    vram = properties.total_memory / 2**30
    print(f"GPU {index}: {properties.name}; VRAM={vram:.2f} GiB")
    if re.search("T4", properties.name, flags=re.IGNORECASE) is None or vram < 14.0:
        raise SystemExit(f"GPU {index} does not satisfy the pinned T4 protocol")
PY

afri-reasoning validate-train \
  --config "${CONFIG_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --validation-file "${VALIDATION_FILE}"

if [[ ! -f "${BASELINE_RUN_DIR}/manifest.json" ]]; then
  echo "Completed baseline not found: ${BASELINE_RUN_DIR}/manifest.json" >&2
  echo "Run and review the zero-shot baseline before any adapter training." >&2
  exit 1
fi

if [[ "${MODE}" == "pipeline-test" ]]; then
  RUN_DIR="${OUTPUT_ROOT}/training-pipeline-test"
  CUDA_VISIBLE_DEVICES=0 afri-train-unsloth \
    --config "${CONFIG_PATH}" \
    --train-file "${TRAIN_FILE}" \
    --validation-file "${VALIDATION_FILE}" \
    --baseline-run-dir "${BASELINE_RUN_DIR}" \
    --output-dir "${RUN_DIR}" \
    --limit-source-groups 5 \
    --max-steps 30 \
    --pipeline-test \
    2>&1 | tee "${OUTPUT_ROOT}/training-pipeline-test.log"
else
  RUN_DIR="${OUTPUT_ROOT}/training-unsloth-qlora-rslora-v1"
  torchrun --standalone --nproc_per_node=2 -m afri_reasoning.train_unsloth \
    --config "${CONFIG_PATH}" \
    --train-file "${TRAIN_FILE}" \
    --validation-file "${VALIDATION_FILE}" \
    --baseline-run-dir "${BASELINE_RUN_DIR}" \
    --output-dir "${RUN_DIR}" \
    --resume-from-checkpoint auto \
    2>&1 | tee "${OUTPUT_ROOT}/training-unsloth-qlora-rslora-v1.log"
fi

RUN_DIR_FOR_CHECK="${RUN_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["RUN_DIR_FOR_CHECK"]) / "training_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "completed":
    raise SystemExit(f"Training did not complete cleanly: {manifest.get('status')}")
print(json.dumps({
    "status": manifest["status"],
    "mode": manifest["mode"],
    "output_dir": str(manifest_path.parent),
    "metrics": manifest.get("metrics"),
}, indent=2))
PY

python -m pip freeze > "${RUN_DIR}/environment.lock.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "${RUN_DIR}/gpu.txt"
echo "Completed ${MODE}: ${RUN_DIR}"
