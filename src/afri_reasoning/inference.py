from __future__ import annotations

import time
from typing import Any


class GemmaThinkingRunner:
    """Single-GPU Gemma 4 runner that preserves parsed thinking and final content."""

    def __init__(self, config: dict[str, Any], *, local_files_only: bool = False) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "Inference requires PyTorch and the baseline dependencies. "
                "Follow docs/HPC_RUNBOOK.md."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. The official baseline must run on an NVIDIA GPU; "
                "do not silently substitute CPU inference."
            )

        self.torch = torch
        self.config = config
        model_config = config["model"]
        set_seed(int(config["generation"]["seed"]))

        self.processor = AutoProcessor.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            local_files_only=local_files_only,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            dtype=model_config.get("dtype", "auto"),
            device_map=model_config.get("device_map", "auto"),
            local_files_only=local_files_only,
        )
        self.model.eval()

    def generate(self, sample: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        inputs = self.processor.apply_chat_template(
            sample["messages"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=bool(self.config["model"]["enable_thinking"]),
        ).to(self.model.device)
        input_length = int(inputs["input_ids"].shape[-1])

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=bool(self.config["generation"]["do_sample"]),
                max_new_tokens=int(self.config["generation"]["max_new_tokens"]),
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        generated_ids = outputs[0][input_length:]
        raw_response = self.processor.decode(generated_ids, skip_special_tokens=False)
        parsed, parse_error = self._parse_response(raw_response, inputs["input_ids"])
        thinking = str(parsed.get("thinking") or "").strip()
        content = str(parsed.get("content") or parsed.get("answer") or "").strip()

        return {
            "sample_id": sample["sample_id"],
            "problem_id": sample["problem_id"],
            "language": sample["language"],
            "language_name": sample["language_name"],
            "question": sample["question"],
            "expected_answer": sample["expected_answer"],
            "source": sample["source"],
            "source_validation": sample["validation"],
            "raw_response": raw_response,
            "thinking": thinking,
            "content": content,
            "parsed_role": parsed.get("role"),
            "parse_valid": parse_error is None and bool(thinking) and bool(content),
            "parse_error": parse_error,
            "input_tokens": input_length,
            "output_tokens": int(generated_ids.shape[-1]),
            "latency_seconds": round(elapsed, 6),
            "tokens_per_second": round(int(generated_ids.shape[-1]) / elapsed, 4),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }

    def _parse_response(self, response: str, prefix: Any) -> tuple[dict[str, Any], str | None]:
        try:
            try:
                parsed = self.processor.parse_response(response, prefix=prefix)
            except TypeError:
                parsed = self.processor.parse_response(response)
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                return {}, f"parse_response returned {type(parsed).__name__}, expected dict"
            return parsed, None
        except Exception as exc:  # parser failures must be saved, not hide the sample
            return {}, f"{type(exc).__name__}: {exc}"
