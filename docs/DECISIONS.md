# Architecture decision record

## ADR-001 - Baseline precedes adaptation

**Decision:** No training code or trained adapter is part of milestone 1.

**Reason:** A frozen “before” measurement is required to demonstrate improvement and prevent post-training benchmark design.

## ADR-002 - Start with Gemma 4 E2B IT

**Decision:** Use `google/gemma-4-E2B-it`, not E4B, for the first complete pipeline.

**Reason:** E2B has native thinking mode and is the lower-risk on-device target. E4B is a later controlled comparison if E2B quality is inadequate.

## ADR-003 - Use matched AfriMGSM questions

**Decision:** Sample the same 50 test-row indices for all four languages.

**Reason:** Pairing reduces differences caused by question difficulty and provides objective final numeric answers.

## ADR-004 - Preserve native Gemma formatting

**Decision:** Enable thinking through `apply_chat_template(..., enable_thinking=True)` and parse with the processor's native response parser.

**Reason:** Hand-written special tokens are fragile and can create a baseline that does not match the model's documented format.

## ADR-005 - Greedy decoding

**Decision:** Use `do_sample=false` for milestone 1.

**Reason:** Deterministic decoding makes paired comparisons and interrupted-run recovery easier to audit. Sampling can be a separate experiment.

## ADR-006 - Human language review is authoritative

**Decision:** Automatic scoring covers numeric correctness and structure. Reasoning quality and language compliance require review.

**Reason:** Low-resource language identification and reasoning judges can produce confident but misleading scores, especially on short or code-switched text.

## ADR-007 - System instruction remains constant in English

**Decision:** The problem is in the target language while the evaluation instruction is a constant English template containing the requested language name.

**Reason:** This isolates question-language behavior while avoiding unverified Wolof meta-instructions. A native-prompt ablation can be added later under a new protocol version.

## ADR-008 - Raw thoughts are research artifacts

**Decision:** Save raw and parsed thoughts locally but ignore run artifacts in Git.

**Reason:** They are needed for the assignment's reasoning evaluation, but they may be large, sensitive, or misleading when published without review.

## ADR-009 - Separate CPU data work from GPU model execution

**Decision:** Use the ICPAC CLIMSA HPC for data preparation, validation, checksums, and
reporting. Run the Gemma 4 baseline on Kaggle T4 x2 unless a Slurm-accessible GPU is
subsequently provisioned.

**Reason:** Audits on the login node and Slurm job 4127 on node05 on 2026-08-21 showed
`Gres=(null)`, no `gres/gpu` in `CfgTRES`, no `nvidia-smi`, and no CUDA toolkit. CPU
capacity does not make adapter training or the official GPU baseline practical. The
alternative two-GPU runner preserves matched problems, provenance, and resumability.

## ADR-010 - Use one advanced primary adapter, then ablate

**Decision:** The first adapted run uses 4-bit QLoRA, rsLoRA, response-only loss, and a
text-only Unsloth adapter. DoRA, NEFTune, preference optimization, reinforcement
learning, synthetic augmentation, and deployment quantization remain separate gated
experiments.

**Reason:** Combining every technique would obscure the cause of both gains and failures.
Low-resource results are more credible when data quality and one controlled intervention
take priority over a long feature list.

## ADR-011 - Treat the first before/after result as a system comparison

**Decision:** Compare the official zero-shot model and adapted 4-bit stack, but do not
attribute the entire delta to training unless a matched Unsloth 4-bit zero-shot control
is also run.

**Reason:** Quantization and the inference runtime change alongside the adapter. A matched
control is required to separate adapter improvement from those effects.
