#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 smoke|full" >&2
  exit 2
fi

: "${PARTITION:?Set PARTITION to the Slurm GPU partition name}"
: "${GRES:?Set GRES, for example gpu:1 or gpu:a100:1}"
: "${VENV_PATH:?Set VENV_PATH to the prepared Python environment}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${PROJECT_ROOT}/logs"

sbatch \
  --partition="${PARTITION}" \
  --gres="${GRES}" \
  --export="ALL,RUN_MODE=${MODE},PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH}" \
  "${PROJECT_ROOT}/slurm/baseline.sbatch"
