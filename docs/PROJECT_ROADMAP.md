# Project roadmap

The roadmap is gated: later work begins only when the preceding evidence is complete.

## Phase 0 - Reproducible foundation

Status: implemented locally; HPC verification pending.

- repository structure and CI;
- pinned experiment configuration;
- data provenance and validation;
- Slurm scripts and hardware audit;
- human-review policy.

## Phase 1 - Zero-shot baseline

Target: week 1.

- audit GPU access;
- prepare 50 matched AfriMGSM problems per language;
- run the four-language smoke test;
- run 200 baseline generations;
- perform native-language review;
- publish a transparent baseline report.

Exit gate: all acceptance criteria in `MILESTONE_01_BASELINE.md`.

## Phase 2 - Twenty-example translation pilot

Target: week 1 after the baseline is running.

- inspect the pinned `HuggingFaceH4/Multilingual-Thinking` schema;
- select 20 source examples using explicit diversity criteria;
- convert Harmony-style fields into a neutral intermediate schema;
- translate prompt, thinking, and final answer into Swahili and Wolof;
- preserve field boundaries and Gemma-native rendering;
- validate all Swahili records;
- obtain native Wolof validation;
- document rejected and corrected examples.

Exit gate: no unverified record is marked training-ready.

## Phase 3 - Curated four-language dataset

Target: weeks 2-4.

- freeze selection and deduplication rules;
- exclude every evaluation problem and near-duplicate;
- balance English, French, Swahili, and Wolof;
- retain translation and verification lineage;
- produce train/validation dataset cards and checksums;
- run format and contamination tests.

Exit gate: signed-off data audit and zero evaluation leakage.

## Phase 4 - One QLoRA method

Target: weeks 4-5.

- freeze Gemma base weights;
- train adapters only;
- overfit 10-20 verified records as a pipeline test;
- run one pilot, one justified correction, and one reproducible final run;
- retain checkpoints, logs, environment, and configuration.

Exit gate: stable loss, valid thinking/final format, and no unexplained language collapse.

## Phase 5 - Paired post-training evaluation

Target: week 6.

- use the exact baseline evaluation file and decoding controls;
- report paired wins, losses, and regressions by problem and language;
- repeat native review;
- inspect English/French regressions and Swahili/Wolof fallback;
- publish limitations and negative results.

Exit gate: measurable improvement without unacceptable language regressions.

## Phase 6 - On-device phase

Target: weeks 7-9, only after the core model passes.

- choose one deployment path (LiteRT or GGUF) based on the target device;
- package/merge the chosen adapter;
- quantize with a recorded method;
- repeat the four-language evaluation;
- measure package size, peak memory, time to first token, decode speed, and thermal behavior;
- verify that thinking survives conversion and quantization.

Exit gate: device-specific evidence, not a theoretical deployment claim.
