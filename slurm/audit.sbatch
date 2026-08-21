#!/usr/bin/env bash
#SBATCH --job-name=gemma4-audit
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/audit-%j.out
#SBATCH --error=logs/audit-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_ROOT}"
bash scripts/hpc/audit_hardware.sh
