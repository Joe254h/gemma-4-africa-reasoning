# Milestone 2 - Human-validated multilingual adaptation

## Objective

Improve reasoning and language adherence in English, French, Swahili, and Wolof using
a reproducible Unsloth QLoRA/rsLoRA run on Kaggle T4 x2, then compare the adapter against
the frozen zero-shot baseline on the same 200 examples.

## Entry gates

- [ ] The 200-example zero-shot baseline is complete and human review has started.
- [ ] Training has exactly 1,200 records: 300 per language.
- [ ] Validation has exactly 120 records: 30 per language.
- [ ] All 330 parallel source groups stay wholly within one split.
- [ ] Swahili and Wolof records have documented native verification.
- [ ] `afri-reasoning validate-train` reports `status: valid`.
- [ ] Kaggle exposes exactly two T4 GPUs.

## Execution gates

- [ ] The pinned environment installs without a dependency conflict.
- [ ] The 30-step one-GPU pipeline test completes with finite loss below the safety cap.
- [ ] The full run launches through `torchrun` with world size two.
- [ ] Checkpoints and the training manifest survive a controlled resume.
- [ ] The final adapter and processor files are present and checksummed.
- [ ] No CUDA out-of-memory, silent CPU fallback, or silent truncation occurs.

## Evaluation gates

- [ ] The adapter runs on the same evaluation checksum as the baseline.
- [ ] All 200 adapted generations exist exactly once.
- [ ] Automatic metrics are computed with the unchanged scorer.
- [ ] Swahili and Wolof reasoning and answer language receive native review.
- [ ] Correctness, language adherence, parse rate, latency, and memory are compared.
- [ ] Confidence intervals or paired bootstrap intervals accompany headline deltas.
- [ ] Regressions and English-fallback examples are included, not hidden.

Training loss alone is not a milestone result. Completion requires the paired evaluation
and review artifacts.

