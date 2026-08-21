# Curated training data contract

Training records are intentionally absent from Git until translation and native review
are complete. Store private working files under `data/train/curated/`; generated model
artifacts belong under `artifacts/`, not here.

Each source problem must form one parallel group containing exactly one English,
French, Swahili, and Wolof record. Keep a whole group in either `train` or `validation`;
never split its language variants across both. The reported v1 corpus has 300 training
groups (1,200 records) and 30 validation groups (120 records).

```json
{
  "schema_version": "1.0",
  "sample_id": "source-0001:sw",
  "source_group_id": "source-0001",
  "language": "sw",
  "split": "train",
  "messages": [
    {"role": "system", "content": "Verified task instruction"},
    {"role": "user", "content": "Verified Swahili prompt"},
    {
      "role": "assistant",
      "reasoning": "Verified Swahili reasoning trace",
      "content": "Verified Swahili final answer"
    }
  ],
  "source": {
    "dataset_id": "HuggingFaceH4/Multilingual-Thinking",
    "revision": "d57770a279da10d30be4845451a499aef8caf5b9",
    "split": "train",
    "source_id": "original-row-id",
    "license": "apache-2.0"
  },
  "validation": {
    "translation_method": "machine-draft-plus-human-revision",
    "training_ready": true,
    "format_valid": true,
    "semantic_accuracy": true,
    "reasoning_consistent": true,
    "language_verified": true,
    "human_verified": true,
    "verifier": "documented-reviewer-id",
    "verification_date": "YYYY-MM-DD",
    "notes": null
  }
}
```

Use `assistant.reasoning`, not hand-written Gemma special tokens. The pinned Gemma 4
chat template renders `reasoning` into its native thought channel and `content` into the
visible answer. The validator rejects reserved format tokens, unverified Swahili/Wolof,
duplicate prompts, imbalanced language counts, cross-split parallel groups, and exact
evaluation leakage. All four variants in a group must share the configured source
dataset, revision, and source row identifier.
