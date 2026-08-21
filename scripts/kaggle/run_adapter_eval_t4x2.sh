#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-configs/baseline.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/gemma4-runs}"
TRAINING_RUN_DIR="${TRAINING_RUN_DIR:-${OUTPUT_ROOT}/training-unsloth-qlora-rslora-v1}"
ADAPTER_DIR="${ADAPTER_DIR:-${TRAINING_RUN_DIR}/adapter}"
KAGGLE_RUN_ID="${KAGGLE_RUN_ID:-adapter-eval-v1}"
RUN_ROOT="${OUTPUT_ROOT}/${KAGGLE_RUN_ID}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}/logs"

TRAINING_MANIFEST="${TRAINING_RUN_DIR}/training_manifest.json" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["TRAINING_MANIFEST"])
manifest = json.loads(path.read_text(encoding="utf-8"))
if manifest.get("status") != "completed":
    raise SystemExit(f"Training is not completed: {path}")
print(f"training_manifest={path}")
PY

afri-reasoning validate-eval --config "${CONFIG_PATH}"

pids=()
for shard_index in 0 1; do
  shard_dir="${RUN_ROOT}/shard-${shard_index}"
  log_path="${RUN_ROOT}/logs/shard-${shard_index}.log"
  echo "Starting adapted-evaluation shard ${shard_index} on T4 ${shard_index}"
  CUDA_VISIBLE_DEVICES="${shard_index}" \
    afri-reasoning run-baseline \
      --config "${CONFIG_PATH}" \
      --adapter-dir "${ADAPTER_DIR}" \
      --run-dir "${shard_dir}" \
      --num-shards 2 \
      --shard-index "${shard_index}" \
      >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for shard_index in 0 1; do
  if wait "${pids[${shard_index}]}"; then
    echo "Adapted-evaluation shard ${shard_index} completed"
  else
    echo "Shard ${shard_index} failed; inspect its log" >&2
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

afri-reasoning merge-baseline \
  --config "${CONFIG_PATH}" \
  --output-dir "${RUN_ROOT}/merged" \
  --shard-dir "${RUN_ROOT}/shard-0" \
  --shard-dir "${RUN_ROOT}/shard-1"

afri-reasoning score --run-dir "${RUN_ROOT}/merged"
python -m pip freeze > "${RUN_ROOT}/merged/environment.lock.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "${RUN_ROOT}/merged/gpu.txt"

echo "Adapted evaluation complete: ${RUN_ROOT}/merged/report.md"
