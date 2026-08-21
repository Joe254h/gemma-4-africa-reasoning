#!/usr/bin/env bash
set -uo pipefail

section() {
  printf '\n### %s\n' "$1"
}

section "Identity and host"
date --iso-8601=seconds 2>/dev/null || date
whoami
hostname -f 2>/dev/null || hostname
pwd

section "Operating system"
uname -a
if [[ -r /etc/os-release ]]; then
  grep -E '^(NAME|VERSION|ID)=' /etc/os-release
fi

section "Slurm overview"
if command -v sinfo >/dev/null 2>&1; then
  sinfo -N -o '%N %P %G %c %m %t'
  printf '\nPartition summary\n'
  sinfo -o '%P %N %G %l %a %t'
else
  echo "sinfo not found"
fi

section "Configured GPU resources"
if command -v scontrol >/dev/null 2>&1; then
  scontrol show nodes | grep -E 'NodeName=|Gres=|CfgTRES=' || true
else
  echo "scontrol not found"
fi

section "Visible NVIDIA devices"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader
else
  echo "nvidia-smi not found on this host"
fi

section "CUDA and compiler tools"
command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc not found"
command -v gcc >/dev/null 2>&1 && gcc --version | head -n 1 || echo "gcc not found"

section "Python"
command -v python3 || true
python3 --version 2>/dev/null || true
python3 - <<'PY' 2>/dev/null || true
try:
    import torch
except ImportError:
    print("torch: not installed in the active Python")
else:
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {properties.name}; "
            f"VRAM={properties.total_memory / 2**30:.2f} GiB"
        )
PY

section "Memory and storage"
free -h 2>/dev/null || true
df -h /scratch 2>/dev/null || df -h .

section "Loaded environment modules"
if command -v module >/dev/null 2>&1; then
  module list 2>&1 || true
else
  echo "environment-modules command not found"
fi

section "Audit conclusion"
echo "For the baseline, repeat this audit inside a Slurm GPU allocation."
echo "Required evidence: an NVIDIA GPU, sufficient free VRAM, and torch CUDA=True."
