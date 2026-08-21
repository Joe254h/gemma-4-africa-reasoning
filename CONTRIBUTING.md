# Contributing

Contributions should strengthen reproducibility, evaluation validity, or African-language quality.

1. Create a focused branch.
2. Explain the experimental assumption being changed.
3. Add or update tests.
4. Run `pytest` and `ruff check .`.
5. Keep evaluation data out of any training path.
6. Never mark Wolof or Swahili text as verified without recording the reviewer and review date.

Do not commit Hugging Face tokens, model weights, caches, raw Slurm logs containing environment secrets, or unreviewed model traces.
