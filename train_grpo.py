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
PROMPT_VARIANTS = (
    "standard",
    "safety_checklist",
    "dominant_signal",
    "brief_chart",
)
PARSE_MODE_REWARD = {
    "exact": 2.0,
    "recovered": 0.75,
    "keyword": -2.0,
    "invalid": -4.0,
}
BENCHMARK_REWARD_SCALE = 24.0
BENCHMARK_REWARD_OFFSET = -6.0
DIFFICULTY_MULTIPLIER = {
    "easy": 1.0,
    "medium": 1.1,
    "hard": 1.25,
}
SAFETY_REWARD_WEIGHT = 0.35
INVALID_SAFETY_PENALTY = -1.5
INVALID_FIELD_PENALTY = -1.5
NON_DIAGNOSE_PENALTY = -5.0
REPEATED_PAIR_PENALTY = -2.0


def _danger_flags(observation: Any) -> list[str]:
    return [flag for flag in (observation.risk_flags or []) if flag.startswith("DANGER")]


def _observation_snapshot(task: dict[str, Any]) -> str:
    observation = task["observation"]
    history_flags = observation.history_flags or ["None"]
    risk_flags = observation.risk_flags or ["None"]
    danger_flags = _danger_flags(observation) or ["None"]
    return (
        f"Case: {task['name']} ({task['difficulty']})\n"
        f"Weeks pregnant: {observation.weeks_pregnant} (trimester {observation.trimester})\n"
        f"Region: {observation.region}\n"
        f"Risk flags: {', '.join(risk_flags)}\n"
        f"Danger flags: {', '.join(danger_flags)}\n"
        f"BP trend: {observation.bp_trend}\n"
        f"Kick count: {observation.avg_kick_count}\n"
        f"Meals/day: {observation.avg_meals}\n"
        f"Sleep hours: {observation.avg_sleep}\n"
        f"Energy: {observation.latest_energy}/10\n"
        f"Breathlessness: {observation.latest_breathlessness}/10\n"
        f"History: {', '.join(history_flags)}\n"
        f"Days of data: {observation.days_of_data}"
    )


def _variant_instructions(prompt_variant: str) -> str:
    if prompt_variant == "safety_checklist":
        return (
            "Checklist before choosing labels:\n"
            "1. If any DANGER_ flag is present, only choose hospital urgency.\n"
            "2. Pick the single dominant condition, not every possible condition.\n"
            "3. History flags raise suspicion, but active danger signals outweigh history.\n"
            "4. If the case is reassuring with no danger signals, avoid over-escalation.\n"
        )
    if prompt_variant == "dominant_signal":
        return (
            "Focus on the dominant signal first:\n"
            "- critical BP or strong preeclampsia symptoms -> preeclampsia\n"
            "- very low kicks -> fetal_distress\n"
            "- bleeding with abdominal pain -> preterm_risk\n"
            "- metabolic strain with diabetes history -> gestational_diabetes\n"
            "- poor nutrition with dizziness / fatigue -> anemia\n"
            "- reassuring pattern with no active warning signs -> low_risk\n"
        )
    if prompt_variant == "brief_chart":
        return (
            "Write the answer as if you are closing a short triage chart note: "
            "one diagnosis, one urgency, one sentence rationale, exact JSON only.\n"
        )
    return (
        "Be clinically decisive and safety-first. Choose one primary condition and one urgency only.\n"
    )


def build_prompt(task: dict[str, Any], prompt_variant: str) -> list[dict[str, str]]:
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
        "This GRPO benchmark scores only the final diagnosis action.\n"
        "Use action_type='diagnose' and keep signal_name set to null.\n"
        "If the observation contains DANGER_ flags, do not under-escalate urgency.\n"
        f"{_variant_instructions(prompt_variant)}\n"
        "Observation snapshot:\n"
        f"{_observation_snapshot(task)}\n\n"
        "Original task prompt:\n"
        f"{task['prompt']()}\n\n"
        "Required reasoning style:\n"
        "- prioritize active danger flags over background history\n"
        "- avoid choosing low_risk when danger flags or strong acute symptoms are present\n"
        "- avoid hospital escalation for clearly reassuring cases with no warning flags\n\n"
        "Return exactly this JSON schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_dataset_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in TASKS:
        for prompt_variant in PROMPT_VARIANTS:
            records.append(
                {
                    "task_id": task["id"],
                    "task_name": task["name"],
                    "difficulty": task["difficulty"],
                    "prompt_variant": prompt_variant,
                    "prompt": build_prompt(task, prompt_variant),
                }
            )
    return records


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


def _normalize_action_payload(payload: dict[str, Any]) -> ActionModel:
    condition = payload.get("condition") or payload.get("target")
    urgency = payload.get("urgency")
    signal_name = payload.get("signal_name")
    action_type = payload.get("action_type")
    if action_type not in {"assess", "request_signal", "diagnose"}:
        if signal_name:
            action_type = "request_signal"
        elif condition or urgency:
            action_type = "diagnose"
        else:
            action_type = None
    return ActionModel(
        condition=condition,
        urgency=urgency,
        rationale=payload.get("rationale"),
        action_type=action_type,
        signal_name=signal_name,
        target=payload.get("target"),
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

    keyword_action = ActionModel(
        condition=_first_label_match(raw_text, VALID_CONDITIONS),
        urgency=_first_label_match(raw_text, VALID_URGENCIES),
        rationale="Keyword-recovered action from non-JSON completion.",
    )
    if keyword_action.condition or keyword_action.urgency:
        keyword_action.action_type = "diagnose"
        return keyword_action, "keyword"

    return ActionModel(rationale="Unparseable completion."), "invalid"


def _benchmark_reward(task: dict[str, Any], action: ActionModel) -> tuple[float, float]:
    grade_result = task["grade"](
        {
            "condition": action.condition,
            "urgency": action.urgency,
            "rationale": action.rationale,
            "target": action.condition,
        }
    )
    benchmark_score = float(grade_result.get("score", 0.0))
    difficulty_multiplier = DIFFICULTY_MULTIPLIER.get(task["difficulty"], 1.0)
    reward = ((benchmark_score * BENCHMARK_REWARD_SCALE) + BENCHMARK_REWARD_OFFSET) * difficulty_multiplier
    return benchmark_score, reward


def _safety_reward(task: dict, action: ActionModel) -> float:
    if action.condition not in VALID_CONDITIONS or action.urgency not in VALID_URGENCIES:
        return INVALID_SAFETY_PENALTY
    breakdown = calculate_reward(action.condition, action.urgency, task["observation"])
    return float(breakdown.reward) * SAFETY_REWARD_WEIGHT


def _field_reward(action: ActionModel) -> float:
    reward = 0.0
    reward += 0.75 if action.condition in VALID_CONDITIONS else INVALID_FIELD_PENALTY
    reward += 0.75 if action.urgency in VALID_URGENCIES else INVALID_FIELD_PENALTY
    return reward


def _clinical_alignment_reward(task: dict[str, Any], action: ActionModel) -> float:
    observation = task["observation"]
    risk_flags = set(observation.risk_flags or [])
    history_flags = set(observation.history_flags or [])
    reward = 0.0

    if any(flag.startswith("DANGER") for flag in risk_flags):
        if action.urgency == "go_to_hospital_today":
            reward += 3.0
        elif action.urgency in VALID_URGENCIES:
            reward -= 8.0
        if action.condition == "low_risk":
            reward -= 6.0

    reassuring_case = (
        not risk_flags
        and observation.avg_kick_count is not None
        and observation.avg_kick_count >= 8
        and (observation.avg_meals or 0) >= 3
        and (observation.avg_sleep or 0) >= 7
        and (observation.latest_energy or 0) >= 6
    )
    if reassuring_case:
        if action.condition == "low_risk" and action.urgency == "monitor_at_home":
            reward += 2.5
        elif action.urgency in {"visit_phc_this_week", "go_to_hospital_today"}:
            reward -= 1.5

    if "DANGER_BP_CRITICAL" in risk_flags or "HIGH_PREECLAMPSIA_SIGNAL" in risk_flags:
        if action.condition == "preeclampsia":
            reward += 1.5
    if "DANGER_LOW_KICKS" in risk_flags or ((observation.avg_kick_count or 99) < 4):
        if action.condition == "fetal_distress":
            reward += 1.5
    if "DANGER_BLEEDING" in risk_flags or "ABDOMINAL_PAIN_SIGNAL" in risk_flags:
        if action.condition == "preterm_risk":
            reward += 1.5
    if "family_diabetes" in history_flags and (observation.avg_meals or 0) > 3.5:
        if action.condition == "gestational_diabetes":
            reward += 1.0
    if "LOW_NUTRITION" in risk_flags or (observation.avg_meals or 99) < 2:
        if action.condition == "anemia":
            reward += 1.0

    return reward


def reward_fn(prompts, completions, task_id, log_extra=None, log_metric=None, **kwargs):
    task_lookup = {task["id"]: task for task in TASKS}
    raw_entries: list[tuple[dict[str, Any], ActionModel, str, float, float, float, float]] = []

    for completion, current_task_id in zip(completions, task_id):
        task = task_lookup[current_task_id]
        raw_text = _completion_text(completion)
        action, parse_mode = _parse_completion_action(raw_text)
        benchmark_score, benchmark_reward = _benchmark_reward(task, action)
        try:
            safety_reward = _safety_reward(task, action)
        except Exception:
            safety_reward = INVALID_SAFETY_PENALTY
        clinical_alignment_reward = _clinical_alignment_reward(task, action)
        raw_entries.append(
            (
                task,
                action,
                parse_mode,
                benchmark_score,
                benchmark_reward,
                safety_reward,
                clinical_alignment_reward,
            )
        )

    pair_counts: dict[tuple[str, str], int] = {}
    for _, action, _, _, _, _, _ in raw_entries:
        pair_key = (action.condition or "", action.urgency or "")
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    rewards: list[float] = []
    parse_modes: list[str] = []
    predicted_conditions: list[str] = []
    predicted_urgencies: list[str] = []
    benchmark_scores: list[float] = []
    safety_rewards: list[float] = []
    clinical_rewards: list[float] = []
    repetition_penalties: list[float] = []

    for task, action, parse_mode, benchmark_score, benchmark_reward, safety_reward, clinical_alignment_reward in raw_entries:
        action_reward = 0.75 if action.action_type == "diagnose" else NON_DIAGNOSE_PENALTY
        format_reward = PARSE_MODE_REWARD[parse_mode]
        field_reward = _field_reward(action)
        pair_key = (action.condition or "", action.urgency or "")
        repetition_penalty = REPEATED_PAIR_PENALTY if pair_counts.get(pair_key, 0) > 1 and pair_key != ("", "") else 0.0
        total_reward = (
            benchmark_reward
            + safety_reward
            + clinical_alignment_reward
            + format_reward
            + action_reward
            + field_reward
            + repetition_penalty
        )

        rewards.append(round(float(total_reward), 4))
        parse_modes.append(parse_mode)
        predicted_conditions.append(action.condition or "")
        predicted_urgencies.append(action.urgency or "")
        benchmark_scores.append(round(benchmark_score, 4))
        safety_rewards.append(round(safety_reward, 4))
        clinical_rewards.append(round(clinical_alignment_reward, 4))
        repetition_penalties.append(round(repetition_penalty, 4))

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
        log_metric("maas/mean_clinical_alignment_reward", sum(clinical_rewards) / total)
        log_metric("maas/mean_repetition_penalty", sum(repetition_penalties) / total)
        log_metric("maas/exact_json_rate", sum(mode == "exact" for mode in parse_modes) / total)
        log_metric(
            "maas/structured_output_rate",
            sum(mode in {"exact", "recovered", "keyword"} for mode in parse_modes) / total,
        )
        unique_pairs = len({(condition, urgency) for condition, urgency in zip(predicted_conditions, predicted_urgencies)})
        log_metric("maas/unique_prediction_pair_rate", unique_pairs / total)
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
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="./artifacts/niva-grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
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
