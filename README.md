# Gemma 4 African Multilingual Reasoning

[![CI](https://github.com/Joe254h/gemma-4-africa-reasoning/actions/workflows/ci.yml/badge.svg)](https://github.com/Joe254h/gemma-4-africa-reasoning/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

A reproducible research project for measuring and improving on-device reasoning in English, French, Swahili, and Wolof. The first milestone is deliberately narrow: establish an auditable **zero-shot Gemma 4 E2B baseline with thinking enabled before any fine-tuning**.

## Why this project matters

Small reasoning models are increasingly deployable on phones and laptops, but benchmark averages hide a practical failure: a model may solve a problem yet abandon the language the user selected. This project therefore separates four questions:

1. Is the final answer correct?
2. Is the reasoning logically correct?
3. Is the reasoning written in the requested language?
4. Is the final answer written in the requested language?

English fallback in Swahili and Wolof is treated as a measured error, not a cosmetic issue.

## Milestone 1: zero-shot baseline

The baseline uses:

- `google/gemma-4-E2B-it`, pinned to a recorded model revision;
- thinking mode enabled through Gemma 4's native chat template;
- 50 matched AfriMGSM test problems in each language (200 generations total);
- deterministic greedy decoding;
- exact numeric accuracy and format checks;
- human review fields for reasoning correctness and language compliance;
- run manifests containing code, model, dataset, package, hardware, and generation provenance.

No LoRA/QLoRA training is included in this milestone. Training begins only after the baseline and evaluation protocol are frozen.

## Repository map

```text
configs/                 Experiment configuration
data/eval/               Generated, provenance-rich evaluation files
docs/                    Protocol, decisions, HPC runbook, and project pitch
scripts/hpc/             Cluster audit, environment, and Slurm submission helpers
slurm/                   Compute-node job definitions
src/afri_reasoning/      Dataset, inference, validation, scoring, and review code
tests/                   Fast tests that do not download a model
artifacts/               Local/HPC run outputs; ignored by Git
```

## Quick start on the HPC

First, create this repository on GitHub as `Joe254h/gemma-4-africa-reasoning`, clone it under `/scratch/njoel`, and enter it. Then:

```bash
bash scripts/hpc/audit_hardware.sh | tee hardware-audit.txt
export VENV_PATH=/scratch/njoel/envs/gemma4-reasoning
bash scripts/hpc/bootstrap_hpc.sh
source "$VENV_PATH/bin/activate"
hf auth login
afri-reasoning prepare-eval --config configs/baseline.yaml
afri-reasoning validate-eval --config configs/baseline.yaml
```

Do not continue until the audit shows an NVIDIA GPU and `torch.cuda.is_available()` is `True`.

Submit one example per language first:

```bash
export PARTITION=parallel
export GRES=gpu:1
export VENV_PATH=/scratch/njoel/envs/gemma4-reasoning
bash scripts/hpc/submit_baseline.sh smoke
```

After inspecting all four smoke-test outputs, submit the full 200-generation baseline:

```bash
bash scripts/hpc/submit_baseline.sh full
```

The exact procedure, expected files, failure checks, and copy-back commands are in [the HPC runbook](docs/HPC_RUNBOOK.md).
The evidence required to reproduce a reported number is defined in [the reproducibility contract](docs/REPRODUCIBILITY.md).

## Local developer checks

The unit tests do not require a GPU or gated model access:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
```

For baseline inference, install the GPU-specific PyTorch build first, then the pinned HPC requirements:

```bash
python -m pip install -r requirements/hpc.txt
python -m pip install -e . --no-deps
```

## Evidence, not placeholders

The repository does not contain invented benchmark numbers. A valid result appears only after a completed run produces:

```text
artifacts/baseline/<run-id>/
├── manifest.json
├── predictions.jsonl
├── automatic_metrics.json
├── human_review.csv
└── report.md
```

`report.md` clearly distinguishes automatic results from human-verified results. The raw artifacts are ignored by Git by default; publish only reviewed aggregate metrics and carefully selected examples.

## Scientific gates

- Baseline before training.
- Evaluation test data never enters training.
- Wolof claims require a named native-language reviewer.
- Swahili claims require native-language review by the project lead or another documented reviewer.
- Reasoning and final-answer language are scored separately.
- Model thoughts are stored for research evaluation but are not reused as multi-turn history.
- Quantization and on-device claims are phase two.

## Primary references

- [Google: Thinking mode in Gemma](https://ai.google.dev/gemma/docs/capabilities/thinking)
- [Google: Gemma 4 prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [Gemma 4 E2B IT model card](https://huggingface.co/google/gemma-4-E2B-it)
- [AfriMGSM dataset card](https://huggingface.co/datasets/masakhane/afrimgsm)
- [Multilingual-Thinking dataset card](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)

## Current status

Milestone 1 implementation is ready for the HPC GPU audit and first smoke run. See [the milestone checklist](docs/MILESTONE_01_BASELINE.md) for the acceptance criteria.
