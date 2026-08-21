# Project pitch

## One sentence

We are building and rigorously evaluating a compact Gemma 4 reasoning model that can think and answer in English, French, Swahili, and Wolof without collapsing back to English.

## Problem

Most multilingual model claims are reported as broad averages. Those averages do not tell a Swahili or Wolof user whether the model can sustain a logical explanation in their language. Low-resource systems can return a correct number while producing unusable or English-only reasoning.

## Technical contribution

- matched four-language reasoning evaluation;
- separate scoring of logical correctness and final correctness;
- separate scoring of thinking language and answer language;
- native-speaker validation gates;
- reproducible model, dataset, hardware, and prompt provenance;
- parameter-efficient adaptation planned only after a frozen baseline;
- on-device measurement planned only after multilingual quality is proven.

## Why it is credible

The project does not start with a fine-tuned demo. It starts with a falsifiable baseline, matched test problems, raw run manifests, and explicit review denominators. It publishes failures and English fallback alongside headline accuracy.

## Portfolio evidence

A strong portfolio version should show:

1. the experiment architecture;
2. a reproducible HPC run;
3. the baseline table with verified denominators;
4. before/after paired evaluation;
5. three carefully explained failures;
6. an interactive or on-device demonstration only after quantitative validation;
7. a concise model card and data card.

## Target audiences

- African-language NLP research groups;
- on-device and edge-AI teams;
- organizations deploying education, agriculture, climate, or public-service assistants;
- employers looking for evidence of ML systems engineering, evaluation discipline, and responsible low-resource NLP.

## Claims we will not make prematurely

- “state of the art” without a benchmark comparison;
- “native quality” without native review;
- “on device” without target-device measurements;
- “improved reasoning” based only on training loss;
- “multilingual” based only on translated final answers.
