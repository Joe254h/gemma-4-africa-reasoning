# Advanced low-resource training stack

This is a controlled research stack for one primary experiment, not a collection of
techniques switched on at the same time. All follow-up ablations remain separate so a
failure or improvement can be traced to a specific intervention.

## Primary Kaggle T4 x2 protocol

| Layer | Pinned choice | Reason |
|---|---|---|
| Base | `unsloth/gemma-4-E2B-it` at an exact revision | Unsloth's current Gemma 4 fixes and memory-efficient kernels |
| Precision | 4-bit QLoRA with NF4/bitsandbytes defaults | Keeps a full model replica inside each 16 GiB T4 |
| Adapter | rank 32, alpha 32, rsLoRA | Stable rank scaling without full-parameter training |
| Scope | language, attention, and MLP modules only | Freezes unused multimodal encoders and avoids wasting T4 memory |
| Objective | native Gemma thought plus final answer; prompt tokens masked | Trains reasoning and answer tokens without learning the user prompt |
| Distributed | PyTorch DDP, two processes, one T4 per process | Uses both GPUs for throughput; it does not combine their VRAM |
| Stability | Unsloth checkpointing, FP16, AdamW 8-bit, gradient clipping | Fits the hardware and guards unstable optimization |
| Selection | validation loss, cosine schedule, early stopping | Reduces overfitting risk on a small curated corpus |
| Evidence | checksums, package/GPU capture, restartable checkpoints | Makes every reported adapter traceable |

The effective global batch is `1 sample x 8 accumulation steps x 2 GPUs = 16`.
With 1,200 training records, one epoch is 75 optimizer steps and the three-epoch cap is
225 steps. Validation contains 120 records. The runner refuses a reported run with a
different corpus size, fewer than two T4 processes, an incomplete baseline, evaluation
leakage, or silent sequence truncation.

`target_modules` is deliberately `null`. In current Unsloth releases, specifying
`all-linear` for a multimodal model also enables the vision/audio side. Allowing Unsloth
to derive modules from the explicit `finetune_*` switches preserves a text-only adapter.

## Data is the main model technology

Each source task is a parallel group with English, French, Swahili, and Wolof variants.
Groups never cross train/validation splits. Every record carries its source revision,
license, validation flags, and reviewer identity. Swahili and Wolof require documented
native review. The validator rejects duplicate normalized prompts, raw Gemma control
tokens, language imbalance, incomplete groups, and exact overlap with the frozen
AfriMGSM evaluation set.

The assistant schema has two distinct fields:

- `reasoning`: rendered into Gemma 4's native thought channel;
- `content`: rendered into the visible final-answer channel.

No benchmark item, automatic translation, or unreviewed synthetic trace becomes
training data simply because it is available.

## Gated follow-up experiments

These are valuable technologies, but they are not enabled in the primary run:

1. Standard LoRA versus rsLoRA ablation, with the same rank and seed.
2. DoRA at lower rank if a memory probe passes; it can consume more memory.
3. NEFTune as one isolated regularization ablation.
4. Continued pretraining on licensed monolingual Swahili/Wolof text before SFT.
5. DPO or ORPO only after native reviewers create chosen/rejected preference pairs.
6. GRPO only for tasks with executable or exact-answer rewards; language reward must
   not be an unreliable automatic proxy for native quality.
7. Back-translation and synthetic reasoning for candidate generation only, followed by
   semantic checks and native review before admission.
8. GGUF or LiteRT quantization only after the adapted model passes the paired evaluation.

Do not stack these into one first experiment. That would make failures hard to diagnose
and improvements impossible to attribute.

The official baseline and adapted stack use different loading paths and precision. The
headline before/after result is therefore a **system-level comparison**, not automatically
a causal estimate of adapter quality. Before claiming that the adapter alone caused a
gain, add an Unsloth 4-bit zero-shot control with the same inference environment; compare
that control to the adapter, and report the official-to-4-bit quantization delta separately.

## Required experiment sequence

```text
frozen evaluation -> official zero-shot baseline -> curate and validate training data
-> optional matched 4-bit control -> 30-step pipeline test -> full two-T4 SFT -> same evaluation
-> native human review -> paired before/after report -> one ablation at a time
```

The primary implementation follows the official documentation for
[Unsloth multi-GPU training](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth),
[TRL supervised fine-tuning](https://huggingface.co/docs/trl/sft_trainer),
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora), and
[Gemma thinking](https://ai.google.dev/gemma/docs/capabilities/thinking).
