# Evaluation protocol v1

This document is frozen before training. Any later change that affects comparability requires a new protocol version and a fresh baseline.

## Research questions

For each language, measure:

1. final numeric correctness;
2. logical correctness of the reasoning trace;
3. presence of a valid thinking channel and final response;
4. thinking-language compliance;
5. final-answer-language compliance;
6. fallback to English;
7. inference latency, output throughput, and peak allocated GPU memory.

## Evaluation set

- Source: `masakhane/afrimgsm`
- Revision: pinned in `configs/baseline.yaml`
- Split: `test`
- Languages/configs: English/`eng`, French/`fra`, Swahili/`swa`, Wolof/`wol`
- Sampling: 50 deterministic indices sampled with seed `20260813`
- Pairing: the same source row index is used in every language
- Integrity check: all four variants must have an identical numeric reference answer

AfriMGSM is used because it provides matched mathematical reasoning questions and objective final answers. It is not a complete measure of real-world reasoning, cultural knowledge, safety, or conversational ability.

## Prompt and decoding controls

- Model: `google/gemma-4-E2B-it`
- Model revision: pinned in configuration
- Native thinking mode: enabled through `enable_thinking=True`
- Decoding: greedy (`do_sample=false`)
- Maximum generated tokens: 2,048
- System instruction: identical except for the requested language name
- User message: the benchmark question without editing

The final response is requested as a short target-language sentence ending in the numeric answer. This makes final-answer language compliance measurable while retaining deterministic numeric extraction.

The system message is in English to hold the instruction constant while the reasoning problem itself is in the target language. This is an explicit design choice and limitation. A later ablation may compare native-language instructions, but it must not replace this baseline silently.

## Automatic metrics

### Final-answer accuracy

The scorer extracts the final number appearing in the parsed final response and compares its normalized decimal value with the reference answer.

### Format validity

A generation is format-valid only when Gemma's parser returns both non-empty `thinking` and final `content` fields.

### Thinking presence

The thinking field must be non-empty. Presence does not imply correctness.

### Performance

Measure wall-clock generation latency, generated tokens per second, and PyTorch peak allocated GPU memory per sample. Slurm queue time is excluded.

## Human metrics

Reviewers fill `human_review.csv` using `true`, `false`, or blank:

- `reasoning_correct`
- `thinking_language_correct`
- `answer_language_correct`
- `thinking_fallback_to_english`
- `answer_fallback_to_english`

Blank means unreviewed, not false. Swahili and Wolof rows are counted as verified only when the reviewer name, review date, and matching native-language code are recorded.

### Reasoning-correct rubric

Mark `true` when the trace uses a valid sequence of steps that supports the final conclusion. Mark `false` for invalid operations, contradictory steps, unjustified leaps that change the result, or reasoning unrelated to the problem. A correct final number reached through invalid reasoning remains `false`.

### Language-compliance rubric

Mark each channel separately. Mathematical symbols, numerals, named entities, and standard technical abbreviations do not count as language violations. Substantial explanatory text in another language does.

### English-fallback rubric

Mark `true` when most or a decisive portion of the relevant channel switches to English for a non-English request. Code-switching can be noted separately without automatically being labelled full fallback.

## Missing and failed outputs

- Never delete a failed generation.
- Parser failures receive `format_valid=false`.
- Job interruption is resumed by `sample_id`; existing rows are skipped.
- Duplicate sample IDs invalidate the run.
- Incomplete native review must be reported with its verified denominator.

## Before/after comparison

A later LoRA/QLoRA result is comparable only when it uses the identical evaluation JSONL, prompt template, model-family chat template, thinking setting, and decoding configuration. Report paired per-problem changes in addition to language averages.
