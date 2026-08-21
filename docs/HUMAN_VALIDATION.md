# Human validation handbook

Human review is a scientific control, not a final proofreading pass.

## Roles

- Swahili reviewer: a native Swahili speaker; the project lead may fill this role.
- Wolof reviewer: a native Wolof speaker identified before results are called verified.
- Reasoning adjudicator: a reviewer capable of checking the mathematics. This can be the language reviewer when qualified.
- Tie-break reviewer: resolves disagreements without seeing the previous reviewer's identity or score when possible.

## Baseline review process

1. Run `afri-reasoning score --run-dir <run-directory>`.
2. Make a working copy of `human_review.csv`.
3. Review the reasoning and final response without changing model output.
4. Enter only `true`, `false`, or blank in Boolean fields.
5. Record reviewer, native-language code, date, and concise notes.
6. Run:

   ```bash
   afri-reasoning score \
     --run-dir <run-directory> \
     --reviews <completed-review.csv>
   ```

7. Inspect `human_metrics.json` and the updated `report.md`.

## Native-language codes

- English: `en`
- French: `fr`
- Swahili: `sw`
- Wolof: `wo`

For Swahili and Wolof, a mismatched or empty `reviewer_native_language` prevents the row from entering verified aggregate metrics.

## Training-data rule for later phases

Each translated record must preserve distinct prompt, thinking, and final-answer fields plus:

- source record ID and checksum;
- translation method and model, if any;
- machine-translated Boolean;
- human-verified Boolean;
- verifier name or controlled identifier;
- verification date;
- semantic-accuracy decision;
- reasoning-consistency decision;
- format-validity decision;
- correction notes;
- split assignment.

No unverified Wolof trace is training-ready. Machine output must never be relabelled as human translation.
