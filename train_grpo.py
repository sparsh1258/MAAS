#!/usr/bin/env python3
"""
Optional GRPO trainer for MAAS.

This script is designed for notebook / Colab / HF Jobs use. It supports:
- plain Transformers + TRL GRPO
- optional Unsloth acceleration via --use-unsloth

It trains directly against the deterministic environment reward.

Each reward_fn call also prints a one-line JSON **reward_probe** (min/mean/max/std)
when there are multiple generations—useful for diagnosing flat `reward_std`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import statistics
import sys
import torch
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ensure_utf8_mode() -> None:
    """
    On some Windows setups, Python defaults to a non-UTF8 locale (e.g. cp1252).
    TRL may read bundled `.jinja` templates without an explicit encoding, which
    can raise UnicodeDecodeError on import. Relaunch in UTF-8 mode before TRL loads.
    """

    if os.name != "nt":
        return
    if sys.flags.utf8_mode:
        return
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])


_ensure_utf8_mode()

from environment import ActionModel
from tasks import TASKS
from xai_reward_model import calculate_reward

VALID_CONDITIONS = (
    "preeclampsia",
    "gestational_diabetes",
    "anemia",
    "preterm_risk",
    "fetal_distress",
    "low_risk",
)
VALID_URGENCIES = (
    "monitor_at_home",
    "visit_phc_this_week",
    "go_to_hospital_today",
)
PARSE_MODE_REWARD = {
    "exact": 2.0,
    "recovered": 0.75,
    "keyword": -2.0,
    "invalid": -4.0,
}
BENCHMARK_REWARD_SCALE = 20.0
BENCHMARK_REWARD_OFFSET = -5.0
SAFETY_REWARD_WEIGHT = 0.35
INVALID_SAFETY_PENALTY = -1.5


def build_prompt(task: dict) -> list[dict[str, str]]:
    schema = {
        "action_type": "diagnose",
        "signal_name": None,
        "condition": "preeclampsia|gestational_diabetes|anemia|preterm_risk|fetal_distress|low_risk",
        "urgency": "monitor_at_home|visit_phc_this_week|go_to_hospital_today",
        "rationale": "short explanation",
    }
    system_prompt = (
        "You are MAAS, a maternal-health triage assistant trained for OpenEnv Theme 3.1. "
        "Be safety-first, reason from observed evidence only, and return exactly one JSON object. "
        "Do not include markdown fences or any text outside the JSON."
    )
    user_prompt = (
        f"{task['prompt']()}\n\n"
        "This GRPO benchmark scores only the final diagnosis action.\n"
        "Use action_type='diagnose' and keep signal_name set to null.\n"
        "If the observation contains DANGER_ flags, do not under-escalate urgency.\n"
        "Return exactly this JSON schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_dataset_records() -> list[dict[str, Any]]:
    return [
        {
            "task_id": task["id"],
            "difficulty": task["difficulty"],
            "prompt": build_prompt(task),
        }
        for task in TASKS
    ]


def _extract_text_blob(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_extract_text_blob(item) for item in value) if part).strip()
    if isinstance(value, dict):
        if "content" in value:
            return _extract_text_blob(value["content"])
        if "text" in value:
            return _extract_text_blob(value["text"])
        if "value" in value:
            return _extract_text_blob(value["value"])
    return str(value)


def _completion_text(completion: Any) -> str:
    return _extract_text_blob(completion).strip()


def _strip_code_fences(raw_text: str) -> str:
    clean = raw_text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(line for line in lines if not line.startswith("```")).strip()
    return clean


def _first_label_match(raw_text: str, labels: tuple[str, ...]) -> str | None:
    lowered = raw_text.lower()
    matches = []
    for label in labels:
        idx = lowered.find(label.lower())
        if idx != -1:
            matches.append((idx, label))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _coerce_action_scalar(value: Any, valid_labels: tuple[str, ...] | None = None) -> str | None:
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if valid_labels is None:
            return candidate
        return _first_label_match(candidate, valid_labels) or candidate
    if isinstance(value, list):
        for item in value:
            coerced = _coerce_action_scalar(item, valid_labels)
            if coerced:
                return coerced
        return None
    return None


def _normalize_action_payload(payload: dict[str, Any]) -> ActionModel:
    # The OpenEnv `ActionModel` in this repo uses `target` for the condition label.
    condition = _coerce_action_scalar(payload.get("condition") or payload.get("target"), VALID_CONDITIONS)
    urgency = _coerce_action_scalar(payload.get("urgency"), VALID_URGENCIES)
    action_type = payload.get("action_type")
    if isinstance(action_type, list):
        action_type = _coerce_action_scalar(action_type)
    if not isinstance(action_type, str) or action_type not in {
        "assess",
        "diagnose",
        "advance_day",
        "request_bp_recheck",
        "request_kick_count",
        "refer_to_phc",
    }:
        action_type = "diagnose" if (condition or urgency) else "assess"
    return ActionModel(
        action_type=action_type or "diagnose",
        target=condition,
        urgency=urgency,
    )


def _parse_completion_action(raw_text: str) -> tuple[ActionModel, str]:
    clean = _strip_code_fences(raw_text)
    candidate: dict[str, Any] | None = None
    parse_mode = "recovered" if clean != raw_text.strip() else "exact"

    try:
        maybe_payload = json.loads(clean)
        if isinstance(maybe_payload, dict):
            candidate = maybe_payload
    except json.JSONDecodeError:
        parse_mode = "recovered" if clean != raw_text.strip() else "invalid"

    if candidate is None:
        match = re.search(r"\{.*\}", clean if parse_mode == "recovered" else raw_text, re.S)
        if match:
            try:
                maybe_payload = json.loads(match.group())
                if isinstance(maybe_payload, dict):
                    candidate = maybe_payload
                    parse_mode = "recovered"
            except json.JSONDecodeError:
                candidate = None

    if candidate is not None:
        return _normalize_action_payload(candidate), parse_mode

    keyword_target = _first_label_match(raw_text, VALID_CONDITIONS)
    keyword_urgency = _first_label_match(raw_text, VALID_URGENCIES)
    if keyword_target or keyword_urgency:
        return (
            ActionModel(
                action_type="diagnose",
                target=keyword_target,
                urgency=keyword_urgency,
            ),
            "keyword",
        )

    return ActionModel(action_type="assess"), "invalid"


def _benchmark_reward(task: dict, action: ActionModel) -> tuple[float, float]:
    grade_result = task["grade"](
        {
            "condition": action.target,
            "urgency": action.urgency,
            "rationale": "",
            "target": action.target,
        }
    )
    benchmark_score = float(grade_result.get("score", 0.0))
    reward = (benchmark_score * BENCHMARK_REWARD_SCALE) + BENCHMARK_REWARD_OFFSET
    return benchmark_score, reward


def _safety_reward(task: dict, action: ActionModel) -> float:
    if action.target not in VALID_CONDITIONS or action.urgency not in VALID_URGENCIES:
        return INVALID_SAFETY_PENALTY
    breakdown = calculate_reward(action.target, action.urgency, task["observation"])
    return float(breakdown.reward) * SAFETY_REWARD_WEIGHT


def reward_fn(prompts, completions, task_id, log_extra=None, log_metric=None, **kwargs):
    task_lookup = {task["id"]: task for task in TASKS}
    rewards: list[float] = []
    parse_modes: list[str] = []
    predicted_conditions: list[str] = []
    predicted_urgencies: list[str] = []
    benchmark_scores: list[float] = []
    safety_rewards: list[float] = []

    for completion, current_task_id in zip(completions, task_id):
        task = task_lookup[current_task_id]
        raw_text = _completion_text(completion)
        action, parse_mode = _parse_completion_action(raw_text)

        benchmark_score, benchmark_reward = _benchmark_reward(task, action)
        try:
            safety_reward = _safety_reward(task, action)
        except Exception:
            safety_reward = INVALID_SAFETY_PENALTY

        action_reward = 0.5 if action.action_type == "diagnose" else (-4.0 if action.action_type else 0.0)
        format_reward = PARSE_MODE_REWARD[parse_mode]
        total_reward = benchmark_reward + safety_reward + format_reward + action_reward

        rewards.append(round(float(total_reward), 4))
        parse_modes.append(parse_mode)
        predicted_conditions.append(action.target or "")
        predicted_urgencies.append(action.urgency or "")
        benchmark_scores.append(round(benchmark_score, 4))
        safety_rewards.append(round(safety_reward, 4))

    if callable(log_extra):
        log_extra("maas_parse_mode", parse_modes)
        log_extra("maas_condition", predicted_conditions)
        log_extra("maas_urgency", predicted_urgencies)
        log_extra("maas_reward", rewards)
        log_extra("maas_benchmark_score", benchmark_scores)
        log_extra("maas_safety_reward", safety_rewards)
    if callable(log_metric) and rewards:
        total = float(len(rewards))
        log_metric("maas/mean_reward", sum(rewards) / total)
        log_metric("maas/mean_benchmark_score", sum(benchmark_scores) / total)
        log_metric("maas/mean_safety_reward", sum(safety_rewards) / total)
        log_metric("maas/exact_json_rate", sum(mode == "exact" for mode in parse_modes) / total)
        log_metric(
            "maas/structured_output_rate",
            sum(mode in {"exact", "recovered", "keyword"} for mode in parse_modes) / total,
        )
    if rewards:
        floats = [float(r) for r in rewards]
        mean = statistics.fmean(floats)
        std = statistics.pstdev(floats) if len(floats) > 1 else 0.0
        print(
            json.dumps(
                {
                    "reward_probe": True,
                    "reward_min": round(min(floats), 4),
                    "reward_mean": round(mean, 4),
                    "reward_max": round(max(floats), 4),
                    "reward_std": round(std, 4),
                    "n": len(floats),
                },
                ensure_ascii=False,
            )
        )
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
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major < 8:
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.is_floating_point():
                    parameter.data = parameter.data.to(torch.float16)
    return model, tokenizer


def save_training_artifacts(output_dir: str, args, train_result, trainer, dataset_size: int) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "dataset_size": dataset_size,
        "prompt_format": "chat_messages",
        "log_completions": args.log_completions,
        "probe_steps": getattr(args, "probe_steps", 0),
        "train_metrics": getattr(train_result, "metrics", {}),
        "log_history": trainer.state.log_history,
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    report = (
        "# MAAS GRPO Run\n\n"
        f"- model: `{args.model_name}`\n"
        f"- epochs: `{args.epochs}`\n"
        f"- batch size: `{args.batch_size}`\n"
        f"- gradient accumulation steps: `{args.gradient_accumulation_steps}`\n"
        f"- num generations: `{args.num_generations}`\n"
        f"- learning rate: `{args.learning_rate}`\n"
        f"- temperature: `{args.temperature}`\n"
        f"- top_p: `{args.top_p}`\n"
        f"- dataset size: `{dataset_size}`\n"
        f"- prompt format: `chat_messages`\n"
        f"- log completions: `{args.log_completions}`\n"
        f"- output dir: `{output_dir}`\n"
        f"- train metrics: `{json.dumps(getattr(train_result, 'metrics', {}), sort_keys=True)}`\n"
    )
    (output_path / "README.md").write_text(report, encoding="utf-8")


def maybe_push_to_hub(output_dir: str, repo_id: str | None, private: bool, commit_message: str) -> None:
    if not repo_id:
        return
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --hub-model-id / --push-to-hub flows."
        ) from exc

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output_dir,
        commit_message=commit_message,
    )
    print(f"Pushed MAAS GRPO artifacts to https://huggingface.co/{repo_id}")


def main(args) -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "GRPO dependencies missing. Install `datasets`, `transformers`, and `trl` first."
        ) from exc

    if args.num_generations < 2:
        raise SystemExit("GRPO requires --num-generations >= 2. Use 2 or more.")

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
        "temperature": args.temperature,
        "top_p": args.top_p,
        "log_completions": args.log_completions,
        "num_completions_to_print": args.num_completions_to_print,
        "use_cpu": args.use_cpu,
    }
    config_signature = inspect.signature(GRPOConfig).parameters
    if "max_prompt_length" in config_signature:
        config_kwargs["max_prompt_length"] = args.max_prompt_length
    train_args = GRPOConfig(**config_kwargs)

    if getattr(args, "probe_steps", 0) and args.probe_steps > 0:
        if "max_steps" in config_signature:
            train_args.max_steps = int(args.probe_steps)
            train_args.num_train_epochs = 1
        else:
            raise SystemExit(
                "This TRL build's GRPOConfig has no max_steps; remove --probe-steps or upgrade TRL."
            )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_training_artifacts(
        output_dir=args.output_dir,
        args=args,
        train_result=train_result,
        trainer=trainer,
        dataset_size=len(dataset),
    )
    repo_id = args.hub_model_id or (os.environ.get("HF_HUB_MODEL_ID") if args.push_to_hub else None)
    if args.push_to_hub and not repo_id:
        raise SystemExit("Use --hub-model-id or set HF_HUB_MODEL_ID when enabling --push-to-hub.")
    maybe_push_to_hub(
        output_dir=args.output_dir,
        repo_id=repo_id,
        private=args.hub_private,
        commit_message=args.hub_commit_message,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional GRPO trainer for the MAAS benchmark.")
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct")

    parser.add_argument("--output-dir", default="./artifacts/niva-grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=0,
        help="If >0, run only this many training steps (sets max_steps) for quick reward-variance checks.",
    )
    parser.add_argument("--num-completions-to-print", type=int, default=4)
    parser.add_argument("--use-unsloth", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--no-log-completions", action="store_false", dest="log_completions")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id")
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-commit-message", default="Upload MAAS GRPO checkpoint")
    parser.set_defaults(log_completions=True)
    main(parser.parse_args())
