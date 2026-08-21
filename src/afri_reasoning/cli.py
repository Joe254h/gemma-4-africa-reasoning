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

    run = subcommands.add_parser("run-baseline", help="Run Gemma 4 thinking inference")
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir")
    run.add_argument("--limit-per-language", type=int)
    run.add_argument("--local-files-only", action="store_true")

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
    elif arguments.command == "run-baseline":
        from afri_reasoning.run import run_baseline

        output = run_baseline(
            load_config(arguments.config),
            run_dir=arguments.run_dir,
            limit_per_language=arguments.limit_per_language,
            local_files_only=arguments.local_files_only,
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
