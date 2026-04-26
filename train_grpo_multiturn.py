#!/usr/bin/env python3
"""GRPO trainer for MAAS multi-turn prenatal trajectories."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch

from environment import ActionModel, MULTITURN_TRAJECTORIES, MultiTurnPrenatalEnvironment
from train_grpo import (
    _completion_text,
    _parse_completion_action,
    load_standard_model,
    load_unsloth_model,
)

SYSTEM_PROMPT = (
    "You are MAAS, a maternal triage assistant operating in a three-day OpenEnv episode. "
    "Reason from visible evidence only, gather more evidence when uncertainty is high, "
    "and return exactly one JSON object. Valid action_type values are "
    "request_bp_recheck, request_kick_count, advance_day, refer_to_phc, and diagnose. "
    "If DANGER flags appear, do not under-escalate urgency."
)

BASELINE_REWARD_SCALE = 10.0
PARSE_MODE_REWARD = {
    "exact": 0.2,
    "recovered": 0.05,
    "keyword": -0.2,
    "invalid": -0.5,
}
ACTION_MATCH_REWARD = 1.5
SAFE_ESCALATION_BONUS = 0.75
CORRECT_TARGET_REWARD = 1.25
CORRECT_URGENCY_REWARD = 1.25
PARTIAL_DIAGNOSIS_REWARD = 0.5

DAY_MESSAGE_ORDERS: list[tuple[str, ...]] = [
    (
        "header",
        "risk_flags",
        "bp_trend",
        "bp",
        "kicks",
        "available_signals",
        "withheld_signals",
        "history",
        "full_text",
    ),
    (
        "header",
        "bp",
        "kicks",
        "risk_flags",
        "available_signals",
        "bp_trend",
        "withheld_signals",
        "history",
        "full_text",
    ),
    (
        "header",
        "available_signals",
        "withheld_signals",
        "bp",
        "kicks",
        "bp_trend",
        "risk_flags",
        "history",
        "full_text",
    ),
]

TRAJECTORY_DIFFICULTY = {
    "traj_low_risk_reassuring": "easy",
    "traj_anemia_gradual": "easy",
    "traj_preeclampsia_slow": "medium",
    "traj_gestational_diabetes_noisy": "medium",
    "traj_preterm_subtle": "medium",
    "traj_fetal_distress_sudden": "hard",
    "traj_mixed_signals_hard": "hard",
    "traj_preeclampsia_fast": "hard",
}
DATASET_RECORDS_CACHE: list[dict[str, Any]] | None = None


def _supports_bfloat16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def _action_to_model(action: dict[str, Any]) -> ActionModel:
    return ActionModel(
        action_type=action["action_type"],
        condition=action.get("condition") or action.get("target"),
        urgency=action.get("urgency"),
        rationale=action.get("rationale"),
        signal_name=action.get("signal_name"),
        target=action.get("target"),
    )


def _ordered_day_message(env: MultiTurnPrenatalEnvironment, order_index: int) -> str:
    observation = env.last_observation or env._build_observation()
    text_observation = env.last_text_observation or env._build_text_observation()
    day = env.current_day
    day_state = env._current_day_state()
    blocks = {
        "header": f"Day {day} observation:",
        "risk_flags": f"- Risk flags: {observation.risk_flags or ['None']}",
        "bp_trend": f"- BP trend: {observation.bp_trend}",
        "bp": f"- Today's blood pressure: {day_state.bp_systolic}/{day_state.bp_diastolic} mmHg",
        "kicks": f"- Today's kick count: {day_state.kick_count}",
        "available_signals": f"- Available signals now: {observation.available_signals}",
        "withheld_signals": f"- Hidden signals still withheld: {observation.withheld_signals or ['None']}",
        "history": (
            f"- History flags: {observation.history_flags or ['hidden']}"
            if day == 3
            else "- History flags: hidden until day 3"
        ),
        "full_text": f"- Full observation text:\n{text_observation}",
    }
    lines = [blocks[key] for key in DAY_MESSAGE_ORDERS[order_index % len(DAY_MESSAGE_ORDERS)]]
    if day == 3:
        lines.append("Make your final diagnosis now unless danger signs require immediate escalation earlier.")
    else:
        lines.append("Choose the safest next action for this day.")
    return "\n".join(lines)


def _has_danger_flags(observation) -> bool:
    return any(str(flag).startswith("DANGER") for flag in (observation.risk_flags or []))


def _has_high_bp(observation) -> bool:
    return "HIGH_BP" in (observation.risk_flags or []) or observation.bp_trend == "rising"


def _has_low_kicks(env: MultiTurnPrenatalEnvironment) -> bool:
    day_state = env._current_day_state()
    return (
        "DANGER_LOW_KICKS" in (env.last_observation.risk_flags if env.last_observation else [])
        or day_state.kick_count < 6
    )


def _teacher_policy_action(env: MultiTurnPrenatalEnvironment) -> dict[str, Any]:
    observation = env.last_observation or env._build_observation()
    if _has_danger_flags(observation):
        return {
            "action_type": "diagnose",
            "target": env.current_trajectory.target_condition,
            "urgency": "go_to_hospital_today",
            "rationale": "Danger flags are visible, so immediate hospital escalation is required.",
        }

    if env.current_day == 1:
        if _has_high_bp(observation) and env.current_day not in env.bp_rechecks:
            return {
                "action_type": "request_bp_recheck",
                "target": None,
                "urgency": None,
                "rationale": "Borderline or rising blood pressure should be confirmed before escalating.",
            }
        if _has_low_kicks(env) and env.current_day not in env.kick_requests:
            return {
                "action_type": "request_kick_count",
                "target": None,
                "urgency": None,
                "rationale": "Low fetal movement should be rechecked before a final triage decision.",
            }
        return {
            "action_type": "advance_day",
            "target": None,
            "urgency": None,
            "rationale": "No immediate danger is visible yet, so gather the next day of evidence.",
        }

    if env.current_day == 2:
        if _has_high_bp(observation) and env.current_day not in env.bp_rechecks:
            return {
                "action_type": "request_bp_recheck",
                "target": None,
                "urgency": None,
                "rationale": "Day-2 blood pressure trend should be confirmed before the final diagnosis.",
            }
        if _has_low_kicks(env) and env.current_day not in env.kick_requests:
            return {
                "action_type": "request_kick_count",
                "target": None,
                "urgency": None,
                "rationale": "Day-2 fetal movement still looks concerning, so confirm the kick count.",
            }
        if observation.risk_flags:
            return {
                "action_type": "refer_to_phc",
                "target": None,
                "urgency": "visit_phc_this_week",
                "rationale": "Persistent non-danger risk on day 2 warrants PHC escalation while gathering final context.",
            }
        return {
            "action_type": "advance_day",
            "target": None,
            "urgency": None,
            "rationale": "Day 3 is needed to resolve the remaining uncertainty safely.",
        }

    return {
        "action_type": "diagnose",
        "target": env.current_trajectory.target_condition,
        "urgency": env.current_trajectory.target_urgency,
        "rationale": "Day 3 provides enough evidence for the final diagnosis.",
    }


def _rollout_teacher_episode(trajectory_id: str, order_index: int) -> dict[str, Any]:
    env = MultiTurnPrenatalEnvironment()
    env.reset(trajectory_id)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_records: list[dict[str, Any]] = []

    while not env.done and env.step_count < env.max_steps:
        user_message = _ordered_day_message(env, order_index)
        messages.append({"role": "user", "content": user_message})
        teacher_action = _teacher_policy_action(env)
        step_result = env.step(_action_to_model(teacher_action))

        turn_records.append(
            {
                "prompt": [dict(message) for message in messages],
                "teacher_action": dict(teacher_action),
                "step_reward": float(step_result.reward),
                "episode_trace": list(step_result.reward_components.get("episode_trace", env.step_logs)),
                "day": env.current_day if not env.done else step_result.observation.days_of_data,
            }
        )
        messages.append({"role": "assistant", "content": json.dumps(teacher_action, ensure_ascii=True)})

    return {
        "trajectory_id": trajectory_id,
        "difficulty": TRAJECTORY_DIFFICULTY.get(trajectory_id, "medium"),
        "order_index": order_index,
        "conversation": messages,
        "turn_records": turn_records,
    }


def _episode_return_from(record: dict[str, Any], step_index: int) -> float:
    remaining = record["turn_records"][step_index:]
    return round(sum(float(turn["step_reward"]) for turn in remaining), 4)


def _build_base_records() -> list[dict[str, Any]]:
    turn_records: list[dict[str, Any]] = []

    for trajectory_id in MULTITURN_TRAJECTORIES:
        for order_index in range(3):
            episode = _rollout_teacher_episode(trajectory_id, order_index)
            for step_index, turn in enumerate(episode["turn_records"]):
                turn_record = {
                    "record_type": "episode",
                    "task_id": f"{trajectory_id}_variant_{order_index + 1}_step_{step_index + 1}",
                    "trajectory_id": trajectory_id,
                    "difficulty": episode["difficulty"],
                    "conversation": episode["conversation"],
                    "prompt": turn["prompt"],
                    "teacher_action": turn["teacher_action"],
                    "teacher_return": _episode_return_from(episode, step_index),
                    "order_index": order_index,
                    "step_index": step_index,
                }
                turn_records.append(turn_record)

                # Hard cases and intermediate evidence-gathering actions get extra weight.
                if episode["difficulty"] == "hard":
                    turn_records.append(dict(turn_record))
                if turn["teacher_action"]["action_type"] in {
                    "request_bp_recheck",
                    "request_kick_count",
                    "refer_to_phc",
                }:
                    turn_records.append(dict(turn_record))

    return turn_records


def _build_trap_examples() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(10):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Day 3 observation:\n"
                    "- Region: Rajasthan\n"
                    "- Weeks pregnant: 24 (Trimester 2)\n"
                    "- Risk flags: []\n"
                    "- BP trend: stable\n"
                    "- Today's blood pressure: 118/76 mmHg\n"
                    "- Today's kick count: 10\n"
                    "- Average kick count across visible days: 10.0\n"
                    "- Average meals across visible days: 3.0\n"
                    "- Average sleep across visible days: 7.5\n"
                    "- History flags: []\n"
                    "Make your final diagnosis now and do not over-escalate."
                ),
            },
        ]
        records.append(
            {
                "record_type": "synthetic",
                "task_id": f"trap_low_risk_{index + 1}",
                "trajectory_id": "trap_low_risk",
                "difficulty": "easy",
                "conversation": prompt,
                "prompt": prompt,
                "teacher_action": {
                    "action_type": "diagnose",
                    "target": "low_risk",
                    "urgency": "monitor_at_home",
                    "rationale": "Reassuring signals should not be escalated.",
                },
                "teacher_return": 1.0,
                "order_index": index % 3,
                "step_index": 0,
            }
        )
    return records


def _build_emergency_examples() -> list[dict[str, Any]]:
    payloads = [
        ("preeclampsia", ["DANGER_BP_CRITICAL", "DANGER_VISION_HEADACHE"]),
        ("fetal_distress", ["DANGER_LOW_KICKS"]),
    ]
    records: list[dict[str, Any]] = []
    for index in range(8):
        condition, flags = payloads[index % len(payloads)]
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Day 1 observation:\n"
                    "- Region: Uttar Pradesh\n"
                    "- Weeks pregnant: 36 (Trimester 3)\n"
                    f"- Risk flags: {flags}\n"
                    "- BP trend: rising\n"
                    "- Today's blood pressure: 162/110 mmHg\n"
                    "- Today's kick count: 2\n"
                    "- Available signals now: ['latest_blood_pressure', 'latest_kick_count']\n"
                    "Danger is already visible on day 1. Choose the safest action now."
                ),
            },
        ]
        records.append(
            {
                "record_type": "synthetic",
                "task_id": f"emergency_{index + 1}",
                "trajectory_id": f"emergency_{index + 1}",
                "difficulty": "hard",
                "conversation": prompt,
                "prompt": prompt,
                "teacher_action": {
                    "action_type": "diagnose",
                    "target": condition,
                    "urgency": "go_to_hospital_today",
                    "rationale": "Visible day-1 danger signs require immediate hospital escalation.",
                },
                "teacher_return": 1.0,
                "order_index": index % 3,
                "step_index": 0,
            }
        )
    return records


def build_dataset_records() -> list[dict[str, Any]]:
    global DATASET_RECORDS_CACHE
    if DATASET_RECORDS_CACHE is not None:
        return DATASET_RECORDS_CACHE

    base_records = _build_base_records()
    trap_records = _build_trap_examples()
    emergency_records = _build_emergency_examples()
    dataset = base_records + trap_records + emergency_records
    random.Random(7).shuffle(dataset)
    DATASET_RECORDS_CACHE = dataset
    return DATASET_RECORDS_CACHE


def _synthetic_reward(record: dict[str, Any], action: ActionModel) -> float:
    expected = record["teacher_action"]
    if action.action_type != expected["action_type"]:
        return 0.0
    if expected["action_type"] != "diagnose":
        return 0.4
    if action.condition == expected["target"] and action.urgency == expected["urgency"]:
        return float(record["teacher_return"])
    if action.condition == expected["target"]:
        return 0.65
    if action.urgency == expected["urgency"]:
        return 0.45
    if action.urgency == "go_to_hospital_today":
        return 0.25
    return 0.0


def _structured_match_reward(
    action: ActionModel,
    expected_action: dict[str, Any],
    episode_return: float,
    current_record_type: str,
    current_record: dict[str, Any] | None,
) -> float:
    reward = 0.0
    expected_type = expected_action.get("action_type")
    predicted_target = action.condition or action.target
    expected_target = expected_action.get("target")
    expected_urgency = expected_action.get("urgency")
    danger_case = current_record_type == "episode" and current_record is not None and (
        TRAJECTORY_DIFFICULTY.get(current_record.get("trajectory_id", ""), "medium") == "hard"
    )

    if action.action_type == expected_type:
        reward += ACTION_MATCH_REWARD

    if expected_type == "diagnose":
        if predicted_target == expected_target:
            reward += CORRECT_TARGET_REWARD
        elif predicted_target in {"preeclampsia", "fetal_distress"} and expected_urgency == "go_to_hospital_today":
            reward += PARTIAL_DIAGNOSIS_REWARD

        if action.urgency == expected_urgency:
            reward += CORRECT_URGENCY_REWARD
        elif action.urgency == "go_to_hospital_today" and expected_urgency == "go_to_hospital_today":
            reward += SAFE_ESCALATION_BONUS

        if danger_case and action.urgency == "go_to_hospital_today":
            reward += SAFE_ESCALATION_BONUS
    else:
        if expected_type == "refer_to_phc" and action.action_type == "diagnose" and action.urgency == "go_to_hospital_today":
            reward += SAFE_ESCALATION_BONUS
        if expected_type in {"request_bp_recheck", "request_kick_count"} and action.action_type == "advance_day":
            reward -= 0.5

    reward += max(0.0, float(episode_return)) * 2.0
    return reward


def multiturn_reward_fn(
    prompts,
    completions,
    trajectory_id,
    record_type,
    teacher_action,
    teacher_return,
    log_extra=None,
    log_metric=None,
    **kwargs,
):
    rewards: list[float] = []
    parse_modes: list[str] = []
    episode_returns: list[float] = []
    teacher_matches: list[int] = []

    dataset_lookup = {
        record["task_id"]: record
        for record in build_dataset_records()
    }
    task_ids = kwargs.get("task_id", [])

    for index, completion in enumerate(completions):
        raw_text = _completion_text(completion)
        action, parse_mode = _parse_completion_action(raw_text)
        parse_modes.append(parse_mode)

        expected_action = teacher_action[index]
        current_record_type = record_type[index]
        current_record = dataset_lookup.get(task_ids[index]) if index < len(task_ids) else None

        if current_record_type == "episode" and current_record is not None:
            env = MultiTurnPrenatalEnvironment()
            env.reset(trajectory_id[index])
            prompt_messages = prompts[index]
            prior_assistant_actions = [
                _parse_completion_action(message["content"])[0]
                for message in prompt_messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            for prior_action in prior_assistant_actions:
                env.step(prior_action)

            try:
                step_result = env.step(action)
                episode_return = float(step_result.reward)
                if not env.done:
                    while not env.done and env.step_count < env.max_steps:
                        follow_up = _teacher_policy_action(env)
                        episode_return += float(env.step(_action_to_model(follow_up)).reward)
            except Exception:
                episode_return = 0.0
        else:
            episode_return = _synthetic_reward(current_record or {"teacher_action": expected_action, "teacher_return": teacher_return[index]}, action)

        exact_match = int(
            action.action_type == expected_action.get("action_type")
            and (action.condition or action.target) == expected_action.get("target")
            and action.urgency == expected_action.get("urgency")
        )
        teacher_matches.append(exact_match)
        episode_returns.append(round(float(episode_return), 4))

        scaled_reward = (
            (episode_return * BASELINE_REWARD_SCALE)
            - 2.0
            + PARSE_MODE_REWARD[parse_mode]
            + _structured_match_reward(
                action,
                expected_action,
                episode_return,
                current_record_type,
                current_record,
            )
            + (0.5 if exact_match else 0.0)
        )
        rewards.append(round(float(scaled_reward), 4))

    if callable(log_extra):
        log_extra("maas_multiturn_parse_mode", parse_modes)
        log_extra("maas_multiturn_episode_return", episode_returns)
        log_extra("maas_multiturn_teacher_match", teacher_matches)
    if callable(log_metric) and rewards:
        total = float(len(rewards))
        log_metric("maas/mean_reward", sum(rewards) / total)
        log_metric("maas/mean_episode_return", sum(episode_returns) / total)
        log_metric("maas/teacher_match_rate", sum(teacher_matches) / total)
        log_metric("maas/exact_json_rate", sum(mode == "exact" for mode in parse_modes) / total)
    return rewards


def _save_summary(output_dir: str, args, train_result, trainer, dataset_records: list[dict[str, Any]]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "training_summary.json"
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "multiturn",
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
        "dataset_size": len(dataset_records),
        "trajectory_count": len(MULTITURN_TRAJECTORIES),
        "prompt_format": "multiturn_conversation_prefixes",
        "dataset_mix": {
            "episode_prefix_examples": sum(record["record_type"] == "episode" for record in dataset_records),
            "trap_examples": sum(record["trajectory_id"] == "trap_low_risk" for record in dataset_records),
            "emergency_examples": sum(record["trajectory_id"].startswith("emergency_") for record in dataset_records),
        },
        "train_metrics": getattr(train_result, "metrics", {}),
        "log_history": trainer.state.log_history,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _save_comparison_plot(multiturn_summary_path: Path, baseline_summary_path: Path | None, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    multiturn_data = json.loads(multiturn_summary_path.read_text(encoding="utf-8"))
    multiturn_history = [row for row in multiturn_data.get("log_history", []) if "step" in row and "reward" in row]
    if not multiturn_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        [row["step"] for row in multiturn_history],
        [row.get("reward") for row in multiturn_history],
        marker="o",
        color="#0f766e",
        label="Multi-turn trained model",
    )

    if baseline_summary_path and baseline_summary_path.exists():
        baseline_data = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        baseline_history = [row for row in baseline_data.get("log_history", []) if "step" in row and "reward" in row]
        if baseline_history:
            ax.plot(
                [row["step"] for row in baseline_history],
                [row.get("reward") for row in baseline_history],
                marker="o",
                color="#2563eb",
                label="Single-step baseline",
            )

    ax.set_title("Single-Step Baseline vs Multi-Turn Reward Curve")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "comparison_reward_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _generate_action_text(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _dialogue_from_model(model, tokenizer, trajectory_id: str, max_new_tokens: int, policy_label: str) -> str:
    env = MultiTurnPrenatalEnvironment()
    env.reset(trajectory_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    transcript = [f"{policy_label} on {trajectory_id}"]

    while not env.done and env.step_count < env.max_steps:
        user_message = _ordered_day_message(env, 0)
        messages.append({"role": "user", "content": user_message})
        transcript.append(f"USER:\n{user_message}")
        raw_action = _generate_action_text(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        transcript.append(f"ASSISTANT:\n{raw_action}")
        action, _ = _parse_completion_action(raw_action)
        try:
            env.step(action)
        except Exception as exc:
            transcript.append(f"[episode halted: {exc}]")
            break
        messages.append({"role": "assistant", "content": raw_action})

    return "\n\n".join(transcript)


def main(args) -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("Missing GRPO dependencies for multi-turn training.") from exc

    if args.num_generations < 2:
        raise SystemExit("GRPO requires --num-generations >= 2.")

    dataset_records = build_dataset_records()
    dataset = Dataset.from_list(dataset_records)

    if args.use_unsloth:
        model, tokenizer = load_unsloth_model(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.no_4bit,
        )
    else:
        model, tokenizer = load_standard_model(args.model_name)

    before_dialogue = _dialogue_from_model(
        model,
        tokenizer,
        "traj_mixed_signals_hard",
        args.max_completion_length,
        "Untrained model",
    )

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
    if not args.use_cpu:
        use_bf16 = _supports_bfloat16()
        config_kwargs["bf16"] = use_bf16
        config_kwargs["fp16"] = not use_bf16
    config_signature = inspect.signature(GRPOConfig).parameters
    if "max_prompt_length" in config_signature:
        config_kwargs["max_prompt_length"] = args.max_prompt_length
    train_args = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=multiturn_reward_fn,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary_path = _save_summary(args.output_dir, args, train_result, trainer, dataset_records)
    _save_comparison_plot(
        summary_path,
        Path(args.baseline_summary) if args.baseline_summary else None,
        args.output_dir,
    )

    after_dialogue = _dialogue_from_model(
        model,
        tokenizer,
        "traj_mixed_signals_hard",
        args.max_completion_length,
        "Trained model",
    )
    comparison_text = before_dialogue + "\n\n" + ("=" * 80) + "\n\n" + after_dialogue
    (Path(args.output_dir) / "mixed_signals_before_after.txt").write_text(
        comparison_text,
        encoding="utf-8",
    )
    print(comparison_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO trainer for the MAAS multi-turn benchmark.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", default="./artifacts/niva-grpo-multiturn")
    parser.add_argument("--baseline-summary", default="./results/grpo_training_summary.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--num-completions-to-print", type=int, default=4)
    parser.add_argument("--use-unsloth", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--no-log-completions", action="store_false", dest="log_completions")
    parser.set_defaults(log_completions=True)
    main(parser.parse_args())
