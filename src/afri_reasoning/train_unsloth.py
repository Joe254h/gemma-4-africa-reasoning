from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afri_reasoning.io import read_json, read_jsonl, sha256_file, write_json_atomic
from afri_reasoning.training_config import load_training_config, training_path
from afri_reasoning.training_data import build_supervised_features, validate_training_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unsloth QLoRA/rsLoRA training for Gemma 4 multilingual reasoning"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-source-groups", type=int)
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        help="Permit a one-GPU, short-run pipeline test; never use for reported results",
    )
    return parser


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str | None]:
    packages = [
        "torch",
        "unsloth",
        "unsloth_zoo",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "huggingface-hub",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validate_baseline(
    directory: str | Path, config: dict[str, Any], evaluation_sha256: str
) -> dict[str, Any]:
    baseline_dir = Path(directory).expanduser().resolve()
    manifest_path = baseline_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed baseline manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise RuntimeError("Training is blocked until the baseline manifest is completed")
    requested = int(manifest.get("requested_samples", -1))
    completed = int(manifest.get("completed_samples", -1))
    if requested != 200 or completed != requested:
        raise RuntimeError(
            f"Training requires the complete 200-sample baseline; requested={requested}, "
            f"completed={completed}"
        )
    if manifest.get("evaluation_sha256") != evaluation_sha256:
        raise RuntimeError("Baseline and training protocol use different evaluation files")
    expected_model = config["model"]["reference_base_id"]
    if manifest.get("model", {}).get("id") != expected_model:
        raise RuntimeError(
            f"Baseline model must be {expected_model}, got {manifest.get('model', {}).get('id')}"
        )
    return {
        "directory": str(baseline_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "run_id": manifest.get("run_id"),
    }


def _limit_groups(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    if limit < 1:
        raise ValueError("limit_source_groups must be at least 1")
    selected_groups: list[str] = []
    for record in records:
        group_id = str(record["source_group_id"])
        if group_id not in selected_groups:
            selected_groups.append(group_id)
        if len(selected_groups) == limit:
            break
    allowed = set(selected_groups)
    return [record for record in records if str(record["source_group_id"]) in allowed]


def _validate_reported_corpus_size(
    config: dict[str, Any],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> None:
    target_train = int(config["dataset"]["target_train_records"])
    validation_fraction = float(config["dataset"]["target_validation_fraction"])
    target_validation = round(target_train * validation_fraction)
    actual = (len(train_records), len(validation_records))
    expected = (target_train, target_validation)
    if actual != expected:
        raise RuntimeError(
            "Reported training requires the frozen corpus size; "
            f"train/validation={actual}, expected={expected}"
        )


def _selection_sha256(records: list[dict[str, Any]]) -> str:
    canonical = "\n".join(str(record["sample_id"]) for record in records)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _resolve_resume(value: str | None, output_dir: Path) -> str | bool | None:
    if value is None:
        return None
    if value != "auto":
        checkpoint = Path(value).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
        return str(checkpoint)
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(output_dir)) or None


def _preflight_torch(config: dict[str, Any], pipeline_test: bool) -> tuple[Any, int, int]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install requirements/kaggle-unsloth.txt") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training must not fall back to CPU")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    required = int(config["runtime"]["required_gpu_count"])
    if pipeline_test:
        if world_size != 1:
            raise RuntimeError("The pipeline test must use exactly one process and one GPU")
    elif world_size != required:
        raise RuntimeError(
            f"Reported training requires torchrun with {required} processes; got {world_size}"
        )

    torch.cuda.set_device(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    expected_pattern = str(config["runtime"]["expected_gpu_name_pattern"])
    if not re.search(expected_pattern, properties.name, flags=re.IGNORECASE):
        raise RuntimeError(
            f"GPU {local_rank} is {properties.name!r}, expected /{expected_pattern}/"
        )
    vram_gib = properties.total_memory / 2**30
    minimum_vram = float(config["runtime"]["minimum_vram_gib"])
    if vram_gib < minimum_vram:
        raise RuntimeError(
            f"GPU {local_rank} has {vram_gib:.2f} GiB, requires at least {minimum_vram:.2f} GiB"
        )
    return torch, local_rank, world_size


def _training_manifest(
    config: dict[str, Any],
    corpus_audit: dict[str, Any],
    baseline: dict[str, Any],
    *,
    output_dir: Path,
    pipeline_test: bool,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    torch: Any,
) -> dict[str, Any]:
    repo_root = Path(config["_meta"]["repo_root"])
    return {
        "schema_version": "1.0",
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "mode": "pipeline-test" if pipeline_test else "reported-ddp-run",
        "project": config["project"],
        "model": config["model"],
        "adapter": config["adapter"],
        "training": config["training"],
        "runtime": config["runtime"],
        "configuration_file": config["_meta"]["config_path"],
        "configuration_sha256": sha256_file(config["_meta"]["config_path"]),
        "corpus": corpus_audit,
        "selection": {
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "train_sample_ids_sha256": _selection_sha256(train_records),
            "validation_sample_ids_sha256": _selection_sha256(validation_records),
        },
        "baseline": baseline,
        "git_commit": _git_commit(repo_root),
        "output_dir": str(output_dir),
        "packages": _package_versions(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "machine": platform.machine(),
        },
        "hardware": {
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_training_config(arguments.config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    corpus_audit = validate_training_corpus(
        config,
        train_file=arguments.train_file,
        validation_file=arguments.validation_file,
    )
    baseline = _validate_baseline(
        arguments.baseline_run_dir,
        config,
        corpus_audit["evaluation_sha256"],
    )
    torch, local_rank, _world_size = _preflight_torch(config, arguments.pipeline_test)
    is_primary = local_rank == 0

    train_path = training_path(config, arguments.train_file or config["paths"]["train_file"])
    validation_path = training_path(
        config, arguments.validation_file or config["paths"]["validation_file"]
    )
    train_records = _limit_groups(read_jsonl(train_path), arguments.limit_source_groups)
    validation_records = read_jsonl(validation_path)
    if not arguments.pipeline_test:
        _validate_reported_corpus_size(config, train_records, validation_records)
    output_dir = training_path(config, arguments.output_dir or config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_manifest.json"
    manifest = _training_manifest(
        config,
        corpus_audit,
        baseline,
        output_dir=output_dir,
        pipeline_test=arguments.pipeline_test,
        train_records=train_records,
        validation_records=validation_records,
        torch=torch,
    )
    if is_primary:
        write_json_atomic(manifest_path, manifest)

    try:
        # Unsloth must patch Transformers before Trainer classes are imported.
        from unsloth import FastVisionModel, get_chat_template  # noqa: I001
        from datasets import Dataset
        from transformers import (
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            TrainerCallback,
        )
        from trl import SFTConfig, SFTTrainer

        model, processor = FastVisionModel.from_pretrained(
            model_name=config["model"]["id"],
            revision=config["model"]["revision"],
            max_seq_length=int(config["model"]["max_seq_length"]),
            load_in_4bit=bool(config["model"]["load_in_4bit"]),
            dtype=torch.float16,
            use_gradient_checkpointing=config["model"]["gradient_checkpointing"],
            device_map={"": local_rank},
        )
        processor = get_chat_template(processor, "gemma-4")
        probe = processor.apply_chat_template(
            [
                {"role": "system", "content": "Reason carefully."},
                {"role": "user", "content": "What is 1 + 1?"},
                {"role": "assistant", "reasoning": "Add the values.", "content": "2"},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if "<|channel>thought" not in probe or "<channel|>" not in probe:
            raise RuntimeError("The loaded chat template did not preserve Gemma thinking format")

        adapter = config["adapter"]
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=bool(adapter["finetune_vision_layers"]),
            finetune_language_layers=bool(adapter["finetune_language_layers"]),
            finetune_attention_modules=bool(adapter["finetune_attention_modules"]),
            finetune_mlp_modules=bool(adapter["finetune_mlp_modules"]),
            r=int(adapter["rank"]),
            lora_alpha=int(adapter["alpha"]),
            lora_dropout=float(adapter["dropout"]),
            bias=str(adapter["bias"]),
            random_state=int(config["training"]["seed"]),
            use_rslora=bool(adapter["use_rslora"]),
            use_dora=bool(adapter["use_dora"]),
            target_modules=adapter["target_modules"],
        )
        FastVisionModel.for_training(model)

        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        maximum_length = int(config["model"]["max_seq_length"])
        train_features = [
            build_supervised_features(record, processor, max_seq_length=maximum_length)
            for record in train_records
        ]
        validation_features = [
            build_supervised_features(record, processor, max_seq_length=maximum_length)
            for record in validation_records
        ]
        train_dataset = Dataset.from_list(train_features)
        validation_dataset = Dataset.from_list(validation_features)
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        training = config["training"]
        report_to = [] if str(training["report_to"]).lower() == "none" else [training["report_to"]]
        max_steps = arguments.max_steps if arguments.max_steps is not None else -1
        trainer_args = SFTConfig(
            output_dir=str(output_dir),
            run_name=config["project"]["experiment"],
            num_train_epochs=float(training["num_train_epochs"]),
            max_steps=max_steps,
            per_device_train_batch_size=int(training["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            warmup_ratio=float(training["warmup_ratio"]),
            lr_scheduler_type=str(training["lr_scheduler_type"]),
            optim=str(training["optim"]),
            max_grad_norm=float(training["max_grad_norm"]),
            logging_steps=int(training["logging_steps"]),
            eval_strategy="steps",
            eval_steps=int(training["eval_steps"]),
            save_strategy="steps",
            save_steps=int(training["save_steps"]),
            save_total_limit=int(training["save_total_limit"]),
            load_best_model_at_end=bool(training["load_best_model_at_end"]),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=int(training["seed"]),
            data_seed=int(training["data_seed"]),
            fp16=bool(training["fp16"]),
            bf16=bool(training["bf16"]),
            tf32=bool(training["tf32"]),
            group_by_length=bool(training["group_by_length"]),
            neftune_noise_alpha=training["neftune_noise_alpha"],
            report_to=report_to,
            gradient_checkpointing=True,
            ddp_backend=str(config["runtime"]["ddp_backend"]),
            ddp_find_unused_parameters=bool(
                config["runtime"]["ddp_find_unused_parameters"]
            ),
            dataloader_num_workers=int(config["runtime"]["dataloader_num_workers"]),
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=maximum_length,
            packing=False,
        )

        class LossGuardCallback(TrainerCallback):
            def on_log(
                self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any
            ) -> None:
                loss = None if logs is None else logs.get("loss")
                if loss is not None and (not math.isfinite(float(loss)) or float(loss) > 50.0):
                    raise RuntimeError(f"Loss safety gate triggered: loss={loss}")

        callbacks = [LossGuardCallback()]
        patience = int(training["early_stopping_patience"])
        if patience > 0 and max_steps < 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        trainer = SFTTrainer(
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
            data_collator=collator,
            callbacks=callbacks,
        )
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        resume = _resolve_resume(arguments.resume_from_checkpoint, output_dir)
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output_dir / "adapter"))
        if trainer.is_world_process_zero():
            processor.save_pretrained(output_dir / "adapter")
            trainer.save_state()
            manifest.update(
                {
                    "status": "completed",
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "adapter_parameters": {
                        "trainable": trainable,
                        "total_loaded": total,
                        "trainable_fraction": trainable / total,
                    },
                    "token_lengths": {
                        "train_min": min(len(item["input_ids"]) for item in train_features),
                        "train_max": max(len(item["input_ids"]) for item in train_features),
                        "validation_min": min(
                            len(item["input_ids"]) for item in validation_features
                        ),
                        "validation_max": max(
                            len(item["input_ids"]) for item in validation_features
                        ),
                    },
                    "resume_from_checkpoint": resume,
                    "metrics": _json_safe(result.metrics),
                    "log_history": _json_safe(trainer.state.log_history),
                }
            )
            write_json_atomic(manifest_path, manifest)
            print(json.dumps({"status": "completed", "output_dir": str(output_dir)}, indent=2))
    except Exception as exc:
        if is_primary:
            manifest["status"] = "failed"
            manifest["failed_at_utc"] = datetime.now(UTC).isoformat()
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            write_json_atomic(manifest_path, manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
