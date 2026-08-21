# Baseline results template

Complete this only with artifacts from a finished run.

## Run identity

- Run ID:
- Git commit:
- Model revision:
- Evaluation SHA-256:
- GPU and VRAM:
- Slurm job ID:

## Automatic results

| Language | N | Final accuracy | Format-valid rate | Thinking-present rate | Mean tok/s |
|---|---:|---:|---:|---:|---:|
| English |  |  |  |  |  |
| French |  |  |  |  |  |
| Swahili |  |  |  |  |  |
| Wolof |  |  |  |  |  |

## Human-verified results

| Language | Verified/N | Reasoning correct | Thinking-language compliance | Answer-language compliance | Answer English fallback |
|---|---:|---:|---:|---:|---:|
| English |  |  |  |  |  |
| French |  |  |  |  |  |
| Swahili |  |  |  |  |  |
| Wolof |  |  |  |  |  |

## Representative errors

Document at least one example from each applicable category:

- correct reasoning and answer;
- correct answer with invalid reasoning;
- incorrect answer with a recoverable arithmetic error;
- valid Wolof/Swahili question with English thinking fallback;
- correct target-language thinking with English final-answer fallback;
- format/parser failure.

## Limitations

- AfriMGSM covers grade-school mathematics, not general reasoning.
- Native-review coverage and reviewer agreement:
- Hardware and latency limitations:
- Prompt-language limitation:
- Any incomplete or failed samples:

## Decision

State whether the baseline is sufficiently complete to begin the 20-example translation pilot. Do not state whether fine-tuning “improves” the model until the post-training paired evaluation exists.
