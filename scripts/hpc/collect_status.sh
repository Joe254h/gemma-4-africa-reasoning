#!/usr/bin/env bash
set -uo pipefail

JOB_ID="${1:-}"
if [[ -z "${JOB_ID}" ]]; then
  echo "Usage: $0 SLURM_JOB_ID" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "Queue status"
squeue -j "${JOB_ID}" -o '%.18i %.12P %.24j %.8u %.2t %.10M %.10l %.6D %R' 2>/dev/null || true

echo
echo "Accounting status"
sacct -j "${JOB_ID}" --format=JobID,JobName,Partition,State,Elapsed,AllocTRES,MaxRSS,ExitCode \
  2>/dev/null || true

echo
echo "Recent output"
tail -n 60 "${PROJECT_ROOT}/logs/baseline-${JOB_ID}.out" 2>/dev/null || true

echo
echo "Recent errors"
tail -n 60 "${PROJECT_ROOT}/logs/baseline-${JOB_ID}.err" 2>/dev/null || true
