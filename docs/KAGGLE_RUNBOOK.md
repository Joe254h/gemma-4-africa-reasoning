# Kaggle T4 x2 runbook

Use this workflow only after the evaluation set is frozen and a one-GPU smoke test has
produced valid Gemma 4 thinking and final-answer fields. Kaggle must be configured with
**GPU T4 x2** and Internet access.

## 1. Prepare access

Accept the `google/gemma-4-E2B-it` terms in the same Hugging Face account used for
Kaggle. Add the account token as a private Kaggle Secret named `HF_TOKEN`. Never paste
the token into a notebook cell, output, dataset, Git commit, or environment lock file.

## 2. Install the pinned project

Run these commands in a Kaggle notebook cell:

```bash
git clone https://github.com/Joe254h/gemma-4-africa-reasoning.git \
  /kaggle/working/gemma-4-africa-reasoning
cd /kaggle/working/gemma-4-africa-reasoning
python -m pip install -r requirements/hpc.txt
python -m pip install -e . --no-deps
```

Then authenticate without printing the token:

```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
```

Use temporary storage for the model cache and persistent working storage for results:

```python
import os

os.environ["HF_HOME"] = "/kaggle/temp/huggingface"
```

## 3. Verify both GPUs

```bash
nvidia-smi -L
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The second command must report `True 2`. Do not launch the dual runner with only one
visible GPU.

## 4. Run and inspect a one-GPU smoke test

The smoke test downloads and caches the pinned model before concurrent workers start:

```bash
cd /kaggle/working/gemma-4-africa-reasoning
CUDA_VISIBLE_DEVICES=0 afri-reasoning run-baseline \
  --config configs/baseline.yaml \
  --run-dir /kaggle/working/gemma4-runs/smoke \
  --limit-per-language 1
afri-reasoning score --run-dir /kaggle/working/gemma4-runs/smoke
```

Inspect all four records and the report before continuing:

```bash
cat /kaggle/working/gemma4-runs/smoke/predictions.jsonl
cat /kaggle/working/gemma4-runs/smoke/report.md
```

The smoke run passes only if all four languages appear, no CUDA out-of-memory error
occurs, and thinking plus final content parse correctly.

## 5. Run the full two-GPU baseline

```bash
cd /kaggle/working/gemma-4-africa-reasoning
export PROJECT_ROOT=/kaggle/working/gemma-4-africa-reasoning
export OUTPUT_ROOT=/kaggle/working/gemma4-runs
export KAGGLE_RUN_ID=baseline-v1
bash scripts/kaggle/run_baseline_t4x2.sh
```

The runner assigns complete matched-problem groups to each T4. Each process writes to
its own resumable directory. The merge step refuses duplicate, missing, wrong-model,
wrong-config, or wrong-evaluation records before scoring.

If Kaggle stops the session, rerun the same command with the same `KAGGLE_RUN_ID` after
restoring `/kaggle/working/gemma4-runs`. Completed sample IDs are skipped. Never combine
shards from different code, model, config, or evaluation revisions.

## 6. Preserve evidence

The completed evidence is located at:

```text
/kaggle/working/gemma4-runs/baseline-v1/merged/
├── automatic_metrics.json
├── environment.lock.txt
├── gpu.txt
├── human_review.csv
├── manifest.json
├── predictions.jsonl
└── report.md
```

Download the entire `baseline-v1` directory or save it as a private Kaggle Notebook
output. Raw reasoning traces require review before public release. Do not commit the
raw run directory directly to Git.

## 7. Prepare the private training dataset

Create one private Kaggle Dataset containing exactly one `train.jsonl` and one
`validation.jsonl`. Follow [the training data contract](../data/train/README.md). The
reported protocol expects 1,200 train records and 120 validation records, balanced
equally across four languages. Do not upload Hugging Face or GitHub tokens with it.

Attach the private dataset to the notebook, then validate it:

```bash
cd /kaggle/working/gemma-4-africa-reasoning
afri-reasoning validate-train \
  --config configs/training/unsloth_qlora_rslora.yaml \
  --train-file /kaggle/input/YOUR-PRIVATE-DATASET/train.jsonl \
  --validation-file /kaggle/input/YOUR-PRIVATE-DATASET/validation.jsonl
```

Stop if any quality gate fails. Fix the data and publish a new private dataset version;
do not weaken the validator to admit a bad record.

## 8. Install the Unsloth environment

Start a fresh **GPU T4 x2** session, authenticate to Hugging Face as in step 2, and run:

```bash
cd /kaggle/working/gemma-4-africa-reasoning
bash scripts/kaggle/bootstrap_unsloth.sh
```

The bootstrap pins Unsloth, TRL, PEFT, Transformers, Accelerate, datasets, and
bitsandbytes and fails on a broken dependency graph or missing CUDA.

## 9. Run the mandatory pipeline test

Restore the completed baseline under
`/kaggle/working/gemma4-runs/baseline-v1/merged`, then run:

```bash
export PROJECT_ROOT=/kaggle/working/gemma-4-africa-reasoning
export OUTPUT_ROOT=/kaggle/working/gemma4-runs
export TRAIN_FILE=/kaggle/input/YOUR-PRIVATE-DATASET/train.jsonl
export VALIDATION_FILE=/kaggle/input/YOUR-PRIVATE-DATASET/validation.jsonl
bash scripts/kaggle/run_training_t4x2.sh pipeline-test
```

This uses five parallel source groups on one T4 for 30 steps. It is an integration test,
not a publishable experiment. Confirm that `training_manifest.json` says `completed`,
loss is finite, both reasoning and final-answer tokens are supervised, and peak memory
leaves headroom.

## 10. Run or resume full two-T4 training

```bash
bash scripts/kaggle/run_training_t4x2.sh full
```

The full command uses `torchrun` with one process per T4. Each T4 holds a model replica;
their 16 GiB memories are not pooled. The effective global batch is 16 and the
three-epoch cap is 225 optimizer steps. `--resume-from-checkpoint auto` resumes the last
valid checkpoint in the same output directory.

Expected output:

```text
/kaggle/working/gemma4-runs/training-unsloth-qlora-rslora-v1/
├── adapter/
├── checkpoint-*/
├── environment.lock.txt
├── gpu.txt
├── trainer_state.json
└── training_manifest.json
```

Save the entire directory as a private notebook output after every session. Never merge
checkpoints from different corpus, config, model, or code checksums.

## 11. Evaluate the adapter on the unchanged benchmark

```bash
bash scripts/kaggle/run_adapter_eval_t4x2.sh
```

This fingerprints the saved adapter, shards the same frozen 200 samples over both T4s,
validates exact coverage, and invokes the unchanged scorer. Preserve the complete
`adapter-eval-v1` directory. Fill its human-review CSV before making language-quality
claims.

## 12. Realistic time budget

These are planning ranges, not benchmark claims. Queueing, first model download, token
length, Kaggle throttling, and reasoning length dominate the variance.

| Stage | Kaggle T4 x2 wall time |
|---|---:|
| Install, authenticate, download, validate | 15-40 min |
| Four-sample baseline smoke | 5-20 min |
| Full 200-sample zero-shot baseline | 2-6 h |
| 30-step training pipeline test | 10-30 min |
| Full 1,200/120-record adapter training | 1-4 h |
| Full 200-sample adapted evaluation | 2-6 h |
| Automatic scoring and packaging | 5-15 min |

Budget two or three Kaggle sessions plus human review. Record actual tokens per second
and elapsed time from the manifests; replace these ranges with measured values only
after the first completed run.
