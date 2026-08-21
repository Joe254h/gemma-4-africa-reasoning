#!/usr/bin/env bash
#SBATCH --job-name=gemma4-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/baseline-%j.out
#SBATCH --error=logs/baseline-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT was not exported by submit_baseline.sh}"
: "${VENV_PATH:?VENV_PATH was not exported by submit_baseline.sh}"
RUN_MODE="${RUN_MODE:-smoke}"
RUN_DIR="${PROJECT_ROOT}/artifacts/baseline/slurm-${SLURM_JOB_ID}-${RUN_MODE}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/${USER}/.cache/huggingface}"

echo "Job ${SLURM_JOB_ID} running on $(hostname) in ${RUN_MODE} mode"
nvidia-smi
python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"CUDA build={torch.version.cuda}")
print(f"CUDA available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the requested Slurm job")
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {properties.name}, {properties.total_memory / 2**30:.2f} GiB")
PY

afri-reasoning validate-eval --config configs/baseline.yaml

RUN_ARGS=(
  run-baseline
  --config configs/baseline.yaml
  --run-dir "${RUN_DIR}"
)
if [[ "${RUN_MODE}" == "smoke" ]]; then
  RUN_ARGS+=(--limit-per-language 1)
fi

srun afri-reasoning "${RUN_ARGS[@]}"
srun afri-reasoning score --run-dir "${RUN_DIR}"
python -m pip freeze > "${RUN_DIR}/environment.lock.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "${RUN_DIR}/gpu.txt"

echo "Completed. Review ${RUN_DIR}/report.md and ${RUN_DIR}/human_review.csv"
