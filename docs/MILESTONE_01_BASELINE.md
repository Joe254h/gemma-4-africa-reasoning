# Milestone 1 - Four-language zero-shot baseline

## Objective

Measure Gemma 4 E2B before any adaptation on matched English, French, Swahili, and Wolof reasoning problems while retaining the model's native thinking output.

## Required evidence

- [x] Login-node and node05 audits confirm that the current Slurm cluster exposes no GPU.
- [ ] Kaggle audit records both T4 models, VRAM, driver, CUDA build, and package versions.
- [ ] `google/gemma-4-E2B-it` loads at the pinned revision.
- [ ] Thinking mode produces separate `thinking` and final `content` fields.
- [ ] A four-sample smoke run completes: one matched problem per language.
- [ ] The full run completes: 50 matched problems per language.
- [ ] Every generation has a manifest-backed provenance record.
- [ ] Automatic numeric accuracy and format metrics are generated.
- [ ] Swahili reasoning and answer language are reviewed by a native Swahili speaker.
- [ ] Wolof reasoning and answer language are reviewed by a native Wolof speaker.
- [ ] Reasoning correctness is reviewed independently of final-answer correctness.
- [ ] Representative failures and English fallback cases are documented.

## Acceptance criteria

The baseline is complete when:

1. all 200 requested generations exist exactly once;
2. no prediction is missing its source `problem_id` or expected answer;
3. parse failures are counted and inspected rather than deleted;
4. language-level automatic metrics are reported;
5. native-reviewed language metrics state both numerator and denominator;
6. the run can be reproduced from a fresh checkout using the recorded revisions;
7. no fine-tuning has used the held-out evaluation records.

This milestone establishes the “before” measurement. It does not claim improvement, training success, or on-device performance.

## Deliverables to show a supervisor

- `hardware-audit.txt` or the Slurm audit log;
- the pinned `configs/baseline.yaml`;
- `manifest.json` from the full run;
- `report.md` before and after human review;
- the completed `human_review.csv`;
- five to ten representative examples covering success, logical failure, format failure, and English fallback;
- one explicit limitation or unanswered question.

## Current unanswered question

Can the ICPAC administrator provide `njoel` access to a separate GPU-enabled Slurm
partition? The current `serial` and `parallel` inventories contain no GPU GRES, so
milestone 1 will use the documented Kaggle T4 x2 path unless cluster access changes.
