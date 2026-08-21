#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-configs/baseline.yaml}"
KAGGLE_RUN_ID="${KAGGLE_RUN_ID:-baseline-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/gemma4-runs}"
RUN_ROOT="${OUTPUT_ROOT}/${KAGGLE_RUN_ID}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}/logs"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Kaggle session")
if torch.cuda.device_count() != 2:
    raise SystemExit(
        f"This runner requires Kaggle T4 x2; detected {torch.cuda.device_count()} GPU(s)"
    )
for index in range(2):
    properties = torch.cuda.get_device_properties(index)
    print(
        f"GPU {index}: {properties.name}; "
        f"VRAM={properties.total_memory / 2**30:.2f} GiB"
    )
PY

afri-reasoning validate-eval --config "${CONFIG_PATH}"

pids=()
for shard_index in 0 1; do
  shard_dir="${RUN_ROOT}/shard-${shard_index}"
  log_path="${RUN_ROOT}/logs/shard-${shard_index}.log"
  echo "Starting shard ${shard_index} on physical GPU ${shard_index}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${shard_index}" \
    afri-reasoning run-baseline \
      --config "${CONFIG_PATH}" \
      --run-dir "${shard_dir}" \
      --num-shards 2 \
      --shard-index "${shard_index}" \
      >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for shard_index in 0 1; do
  if wait "${pids[${shard_index}]}"; then
    echo "Shard ${shard_index} completed"
  else
    echo "Shard ${shard_index} failed; inspect ${RUN_ROOT}/logs/shard-${shard_index}.log" >&2
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

echo "Baseline complete: ${RUN_ROOT}/merged/report.md"
