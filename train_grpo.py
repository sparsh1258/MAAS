#!/usr/bin/env python3
"""
Optional GRPO trainer for MAAS.

This script is designed for notebook / Colab / HF Jobs use. It supports:
- plain Transformers + TRL GRPO
- optional Unsloth acceleration via --use-unsloth

It trains directly against the deterministic environment reward.
"""

from __future__ import annotations

import argparse
import inspect
import json
from typing import Any

from environment import parse_llm_output
from tasks import TASKS
from xai_reward_model import calculate_reward


def build_prompt(task: dict) -> str:
    schema = {
        "action_type": "diagnose",
        "signal_name": None,
        "condition": "preeclampsia|gestational_diabetes|anemia|preterm_risk|fetal_distress|low_risk",
        "urgency": "monitor_at_home|visit_phc_this_week|go_to_hospital_today",
        "rationale": "short explanation",
    }
    return (
        "You are a maternal-health triage assistant.\n"
        "This GRPO benchmark scores the final diagnosis only, so respond with action_type='diagnose'.\n"
        "Read the patient observation below and respond with JSON only.\n"
        f"JSON schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"{task['prompt']()}\n"
    )


def build_dataset_records() -> list[dict[str, str]]:
    return [
        {
            "task_id": task["id"],
            "difficulty": task["difficulty"],
            "prompt": build_prompt(task),
        }
        for task in TASKS
    ]


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("content", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(completion)


def reward_fn(prompts, completions, task_id, **kwargs):
    task_lookup = {task["id"]: task for task in TASKS}
    rewards: list[float] = []

    for completion, current_task_id in zip(completions, task_id):
        task = task_lookup[current_task_id]
        raw_text = _completion_text(completion)
        try:
            action = parse_llm_output(raw_text)
            if action.action_type not in {None, "diagnose"}:
                rewards.append(-20.0)
                continue
            breakdown = calculate_reward(action.condition, action.urgency, task["observation"])
            rewards.append(float(breakdown.reward))
        except Exception:
            rewards.append(-20.0)
    return rewards


def load_standard_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def load_unsloth_model(model_name: str, max_seq_length: int, load_in_4bit: bool):
    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError as exc:
        raise SystemExit(
            "Unsloth is not installed. Install `unsloth` and rerun with --use-unsloth."
        ) from exc

    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def main(args) -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "GRPO dependencies missing. Install `datasets`, `transformers`, and `trl` first."
        ) from exc

    if args.use_unsloth:
        model, tokenizer = load_unsloth_model(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.no_4bit,
        )
    else:
        model, tokenizer = load_standard_model(args.model_name)

    dataset = Dataset.from_list(build_dataset_records())
    config_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "use_cpu": args.use_cpu,
    }
    config_signature = inspect.signature(GRPOConfig).parameters
    if "max_prompt_length" in config_signature:
        config_kwargs["max_prompt_length"] = args.max_prompt_length
    train_args = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional GRPO trainer for the MAAS benchmark.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="./artifacts/niva-grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--use-unsloth", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    main(parser.parse_args())
