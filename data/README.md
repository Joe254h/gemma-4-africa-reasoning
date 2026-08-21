# Data policy

This repository keeps evaluation and future training data separate.

- `data/eval/` is reserved for held-out benchmark data and must never be used for training.
- Future translated training examples must live outside `data/eval/` and include machine-translation and human-verification metadata.
- No Wolof record may be marked training-ready without native-speaker verification.
- Generated files must retain source dataset name, revision, split, row index, and license information.

The evaluation preparation command is deterministic and checks that matched language versions share the same reference answer.
