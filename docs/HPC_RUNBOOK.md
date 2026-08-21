# ICPAC CLIMSA HPC runbook

This runbook assumes login `njoel@197.248.214.27` and a project checkout under `/scratch/njoel`. Commands that use GPU compute must run through Slurm, not on the head node.

## 1. Publish and clone the repository

Create the empty GitHub repository `Joe254h/gemma-4-africa-reasoning`, then from your local project folder:

```bash
git init -b main
git add .
git commit -m "feat: add reproducible Gemma 4 multilingual baseline"
git remote add origin https://github.com/Joe254h/gemma-4-africa-reasoning.git
git push -u origin main
```

On the HPC:

```bash
ssh njoel@197.248.214.27
cd /scratch/njoel
git clone https://github.com/Joe254h/gemma-4-africa-reasoning.git
cd gemma-4-africa-reasoning
```

## 2. Audit available hardware

The login-node audit shows Slurm's view of the cluster:

```bash
bash scripts/hpc/audit_hardware.sh | tee hardware-audit-login.txt
```

Use the output to identify the GPU partition and GRES syntax. Then run the audit on a compute node, replacing the examples with values actually reported by `sinfo`:

```bash
mkdir -p logs
export PROJECT_ROOT=/scratch/njoel/gemma-4-africa-reasoning
sbatch \
  --partition=YOUR_GPU_PARTITION \
  --gres=YOUR_GPU_GRES \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT" \
  slurm/audit.sbatch
```

Examples of possible GRES values are `gpu:1`, `gpu:t4:1`, or `gpu:a100:1`; do not guess. Monitor with:

```bash
squeue -u "$USER"
cat logs/audit-JOB_ID.out
cat logs/audit-JOB_ID.err
```

Stop here if the compute-node log does not show an NVIDIA GPU.

If every node reports `Gres=(null)`, `CfgTRES` contains no `gres/gpu`, and a
compute-node audit cannot find `nvidia-smi`, the cluster has no Slurm-accessible GPU.
Do not submit the baseline job or attempt QLoRA on CPU. Use the cluster for dataset
preparation, validation, checksums, and reporting, then use the documented Kaggle T4 x2
workflow for model inference and training experiments.

## 3. Create the environment

If your cluster uses environment modules, load the recommended Python/CUDA module first. Then:

```bash
cd /scratch/njoel/gemma-4-africa-reasoning
export VENV_PATH=/scratch/njoel/envs/gemma4-reasoning
bash scripts/hpc/bootstrap_hpc.sh
source "$VENV_PATH/bin/activate"
```

If the installed PyTorch CUDA build is incompatible with the driver, remove only this dedicated virtual environment and recreate it with the PyTorch index URL recommended for the cluster:

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cuYOUR_VERSION
bash scripts/hpc/bootstrap_hpc.sh
```

Do not copy that example literally; use the CUDA wheel documented for the HPC driver.

## 4. Authenticate to Hugging Face

Gemma access may require accepting the model terms in your Hugging Face account. Authenticate interactively on the HPC:

```bash
source /scratch/njoel/envs/gemma4-reasoning/bin/activate
hf auth login
hf auth whoami
```

Never place the token in Git, `.env.example`, a Slurm script, or a command pasted into a public issue.

Set the shared cache location:

```bash
export HF_HOME=/scratch/njoel/.cache/huggingface
mkdir -p "$HF_HOME"
```

## 5. Build and freeze the evaluation set

```bash
afri-reasoning prepare-eval --config configs/baseline.yaml
afri-reasoning validate-eval --config configs/baseline.yaml
git status --short
```

If the Hugging Face `datasets` loader is unavailable, check out the exact pinned dataset revision and use the dependency-light TSV reader:

```bash
git clone https://huggingface.co/datasets/masakhane/afrimgsm /scratch/njoel/afrimgsm
git -C /scratch/njoel/afrimgsm checkout 8e4268d2b94941f18f63f694cf48e4ae26fbec65
afri-reasoning prepare-eval \
  --config configs/baseline.yaml \
  --source-dir /scratch/njoel/afrimgsm
```

The command should report 200 records, 50 per language, and 50 matched problems. Commit the generated evaluation JSONL and manifest so future training comparisons use exactly the same set.

## 6. Smoke test

Set the values discovered in the audit:

```bash
export PROJECT_ROOT=/scratch/njoel/gemma-4-africa-reasoning
export VENV_PATH=/scratch/njoel/envs/gemma4-reasoning
export PARTITION=YOUR_GPU_PARTITION
export GRES=YOUR_GPU_GRES
bash scripts/hpc/submit_baseline.sh smoke
```

Record the returned job ID and monitor it:

```bash
bash scripts/hpc/collect_status.sh JOB_ID
```

Inspect all four predictions:

```bash
less artifacts/baseline/slurm-JOB_ID-smoke/predictions.jsonl
cat artifacts/baseline/slurm-JOB_ID-smoke/report.md
```

The smoke test passes only when:

- exactly four predictions exist;
- every language is present once;
- no CUDA out-of-memory error occurred;
- parsed thinking and final content are both present or any parser defect is understood;
- the model revision in `manifest.json` matches the configuration.

## 7. Full baseline

```bash
bash scripts/hpc/submit_baseline.sh full
```

Monitor without repeatedly loading the full log:

```bash
bash scripts/hpc/collect_status.sh JOB_ID
```

The job writes to:

```text
artifacts/baseline/slurm-JOB_ID-full/
```

The prediction writer flushes each record, and a restarted command skips completed `sample_id` values in the same run directory.

## 8. Human review and scoring

Copy the run directory to a secure local working location or review it on the HPC. Fill the generated CSV according to `docs/HUMAN_VALIDATION.md`, then:

```bash
afri-reasoning score \
  --run-dir artifacts/baseline/slurm-JOB_ID-full \
  --reviews artifacts/baseline/slurm-JOB_ID-full/human_review.completed.csv
```

Do not publish raw traces automatically. Commit a reviewed report and selected examples only after checking for personal data, unsafe content, and licensing constraints.

## 9. Copy results back

From your local computer:

```bash
scp -r \
  njoel@197.248.214.27:/scratch/njoel/gemma-4-africa-reasoning/artifacts/baseline/slurm-JOB_ID-full \
  ./baseline-results/
```

## Common failures

### `CUDA is not available`

The job did not receive a GPU, the CUDA/PyTorch builds are incompatible, or the partition/GRES request is wrong. Read the compute-node audit; do not run the baseline on CPU.

### Hugging Face 401 or 403

Accept the model terms on the Hugging Face website and rerun `hf auth login` under the same Unix account used by Slurm.

### CUDA out of memory

Confirm no other process is using the allocated device. The baseline is batch size one. If E2B still cannot load, record the GPU and VRAM and request a larger device rather than changing the model silently.

### Parser returns no thinking field

Confirm Transformers is at least 5.10.1, `enable_thinking` is true, special tokens were not skipped before parsing, and the pinned model revision is loaded.

### Job reaches time limit

Use the same run directory and resubmit with an approved longer wall-time. Existing completed samples are skipped.
