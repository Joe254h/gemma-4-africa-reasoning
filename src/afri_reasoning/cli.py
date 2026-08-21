from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from afri_reasoning.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afri-reasoning",
        description="Reproducible Gemma 4 four-language reasoning baseline",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare-eval", help="Build matched AfriMGSM JSONL")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument(
        "--source-dir",
        help="Optional checkout of the pinned AfriMGSM repository (avoids datasets dependency)",
    )

    validate = subcommands.add_parser("validate-eval", help="Validate evaluation invariants")
    validate.add_argument("--config", required=True)

    validate_train = subcommands.add_parser(
        "validate-train", help="Validate curated four-language training data"
    )
    validate_train.add_argument("--config", required=True)
    validate_train.add_argument("--train-file")
    validate_train.add_argument("--validation-file")

    run = subcommands.add_parser("run-baseline", help="Run Gemma 4 thinking inference")
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir")
    run.add_argument("--limit-per-language", type=int)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument(
        "--adapter-dir",
        help="Evaluate a completed local Unsloth adapter instead of the zero-shot model",
    )

    merge = subcommands.add_parser(
        "merge-baseline", help="Validate and merge completed baseline shards"
    )
    merge.add_argument("--config", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--shard-dir", action="append", required=True)

    score = subcommands.add_parser("score", help="Score predictions and create review queue")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--reviews")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare-eval":
        from afri_reasoning.prepare import prepare_evaluation

        result = prepare_evaluation(
            load_config(arguments.config),
            force=arguments.force,
            source_dir=arguments.source_dir,
        )
    elif arguments.command == "validate-eval":
        from afri_reasoning.validation import validate_evaluation

        result = validate_evaluation(load_config(arguments.config))
    elif arguments.command == "validate-train":
        from afri_reasoning.training_config import load_training_config
        from afri_reasoning.training_data import validate_training_corpus

        result = validate_training_corpus(
            load_training_config(arguments.config),
            train_file=arguments.train_file,
            validation_file=arguments.validation_file,
        )
    elif arguments.command == "run-baseline":
        from afri_reasoning.run import run_baseline

        output = run_baseline(
            load_config(arguments.config),
            run_dir=arguments.run_dir,
            limit_per_language=arguments.limit_per_language,
            local_files_only=arguments.local_files_only,
            num_shards=arguments.num_shards,
            shard_index=arguments.shard_index,
            adapter_dir=arguments.adapter_dir,
        )
        result = {"run_dir": str(output), "status": "completed"}
    elif arguments.command == "merge-baseline":
        from afri_reasoning.merge import merge_baseline_shards

        output = merge_baseline_shards(
            load_config(arguments.config),
            shard_dirs=arguments.shard_dir,
            output_dir=arguments.output_dir,
        )
        result = {"run_dir": str(output), "status": "completed"}
    elif arguments.command == "score":
        from afri_reasoning.scoring import score_run

        result = score_run(arguments.run_dir, arguments.reviews)
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(f"Unhandled command: {arguments.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
