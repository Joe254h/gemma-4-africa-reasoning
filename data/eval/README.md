# Held-out evaluation data

Run:

```bash
afri-reasoning prepare-eval --config configs/baseline.yaml
afri-reasoning validate-eval --config configs/baseline.yaml
```

This creates `afrimgsm_50x4.jsonl` and its manifest from a pinned AfriMGSM revision. The 50 source indices are sampled once with the configured seed and reused in all four languages.

The benchmark's published translation provenance does not replace project-level review. Swahili and Wolof records remain flagged for native review in the generated metadata.
