from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import MULTITURN_TRAJECTORIES, MultiTurnPrenatalEnvironment  # noqa: E402
from xai_reward_model import SAFE_CONDITIONS, URGENCY_ORDER  # noqa: E402


Action = dict[str, Any]


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model response.")


@dataclass(frozen=True)
class EvalRow:
    trajectory_id: str
    policy: str
    predicted_condition: str | None
    predicted_urgency: str | None
    reference_condition: str
    reference_urgency: str
    reward: float
    condition_correct: bool
    urgency_correct: bool
    under_escalated: bool
    over_escalated: bool
    json_valid: bool
    steps: int


def _diagnose(condition: str, urgency: str, rationale: str) -> Action:
    return {
        "action_type": "diagnose",
        "target": condition,
        "urgency": urgency,
        "rationale": rationale,
    }


def oracle_policy(env: MultiTurnPrenatalEnvironment) -> list[Action]:
    assert env.current_trajectory is not None
    return [
        {"action_type": "advance_day", "rationale": "Collect day 2 symptoms before final triage."},
        {"action_type": "advance_day", "rationale": "Collect day 3 history/context before final triage."},
        _diagnose(
            env.current_trajectory.target_condition,
            env.current_trajectory.target_urgency,
            "Oracle benchmark action using the hidden reference label.",
        ),
    ]


def conservative_policy(env: MultiTurnPrenatalEnvironment) -> list[Action]:
    actions: list[Action] = [
        {"action_type": "advance_day", "rationale": "Wait for symptom evidence before final triage."},
        {"action_type": "advance_day", "rationale": "Wait for history/context before final triage."},
    ]
    assert env.current_trajectory is not None
    final_day = env.current_trajectory.days[-1]
    symptoms = final_day.symptoms
    avg_meals = sum(day.meals_count for day in env.current_trajectory.days) / len(env.current_trajectory.days)
    has_diabetes_history = "family_diabetes" in set(env.current_trajectory.history_flags)
    if final_day.kick_count < 3:
        condition = "fetal_distress"
        urgency = "go_to_hospital_today"
    elif final_day.bp_systolic >= 160 or final_day.bp_diastolic >= 110 or (
        symptoms.get("headache") and (symptoms.get("blurred_vision") or symptoms.get("swelling"))
    ):
        condition = "preeclampsia"
        urgency = "go_to_hospital_today"
    elif symptoms.get("cramps") or symptoms.get("bleeding"):
        condition = "preterm_risk"
        urgency = "visit_phc_this_week"
    elif has_diabetes_history and symptoms.get("breathlessness"):
        condition = "gestational_diabetes"
        urgency = "visit_phc_this_week"
    elif avg_meals < 2 or symptoms.get("dizziness"):
        condition = "anemia"
        urgency = "visit_phc_this_week"
    else:
        condition = "low_risk"
        urgency = "monitor_at_home"
    actions.append(_diagnose(condition, urgency, "Deterministic visible-signal baseline."))
    return actions


def day1_fast_policy(env: MultiTurnPrenatalEnvironment) -> list[Action]:
    obs = env.last_prompt.observation if env.last_prompt else env._build_prompt().observation
    flags = set(obs.risk_flags or [])
    if "DANGER_LOW_KICKS" in flags:
        return [_diagnose("fetal_distress", "go_to_hospital_today", "Immediate day-1 danger escalation.")]
    if "DANGER_BP_CRITICAL" in flags:
        return [_diagnose("preeclampsia", "go_to_hospital_today", "Immediate day-1 critical BP escalation.")]
    return [_diagnose("low_risk", "monitor_at_home", "Naive day-1 baseline.")]


def _load_prediction_actions(path: Path) -> dict[str, list[Action]]:
    predictions: dict[str, list[Action]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            trajectory_id = item.get("trajectory_id")
            if not trajectory_id:
                raise ValueError(f"Missing trajectory_id on line {line_number}")
            raw_actions = item.get("actions")
            if raw_actions is None:
                raw_actions = [item.get("action", item)]
            if not isinstance(raw_actions, list):
                raise ValueError(f"actions must be a list on line {line_number}")
            predictions[str(trajectory_id)] = [dict(action) for action in raw_actions]
    return predictions


def _model_action(client: Any, model_id: str, env: MultiTurnPrenatalEnvironment) -> Action:
    prompt = env.last_prompt or env._build_prompt()
    messages = [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": prompt.user_prompt},
    ]
    response = client.chat_completion(
        messages=messages,
        model=model_id,
        max_tokens=220,
        temperature=0.0,
    )
    message = response.choices[0].message
    content = message.get("content") if isinstance(message, dict) else message.content
    parsed = _extract_json_object(str(content))
    return {
        "action_type": parsed.get("action_type") or "diagnose",
        "target": parsed.get("target") or parsed.get("condition"),
        "urgency": parsed.get("urgency"),
        "rationale": parsed.get("rationale") or f"Model action from {model_id}.",
    }


def _run_hf_model(trajectory_id: str, policy_name: str, model_id: str, token: str | None) -> EvalRow:
    try:
        from huggingface_hub import InferenceClient
    except Exception:
        return _run_actions(trajectory_id, policy_name, [])

    env = MultiTurnPrenatalEnvironment()
    env.reset(trajectory_id)
    client = InferenceClient(token=token)
    actions: list[Action] = []
    json_valid = True
    result = None
    for _ in range(env.max_steps):
        try:
            action = _model_action(client, model_id, env)
            actions.append(action)
            result = env.step(action)
        except Exception:
            json_valid = False
            break
        if result.done:
            break
    row = _run_actions(trajectory_id, policy_name, actions)
    if not json_valid:
        return EvalRow(
            trajectory_id=row.trajectory_id,
            policy=row.policy,
            predicted_condition=row.predicted_condition,
            predicted_urgency=row.predicted_urgency,
            reference_condition=row.reference_condition,
            reference_urgency=row.reference_urgency,
            reward=row.reward,
            condition_correct=row.condition_correct,
            urgency_correct=row.urgency_correct,
            under_escalated=row.under_escalated,
            over_escalated=row.over_escalated,
            json_valid=False,
            steps=row.steps,
        )
    return row


def _run_actions(trajectory_id: str, policy_name: str, actions: list[Action]) -> EvalRow:
    env = MultiTurnPrenatalEnvironment()
    env.reset(trajectory_id)
    result = None
    json_valid = True
    for action in actions:
        try:
            result = env.step(action)
        except Exception:
            json_valid = False
            break
        if result.done:
            break

    if result is None or not result.done:
        try:
            result = env.step(_diagnose("low_risk", "monitor_at_home", "Fallback diagnosis after invalid or incomplete action trace."))
        except Exception:
            ref = MULTITURN_TRAJECTORIES[trajectory_id]
            return EvalRow(
                trajectory_id=trajectory_id,
                policy=policy_name,
                predicted_condition=None,
                predicted_urgency=None,
                reference_condition=ref.target_condition,
                reference_urgency=ref.target_urgency,
                reward=0.0,
                condition_correct=False,
                urgency_correct=False,
                under_escalated=ref.target_urgency == "go_to_hospital_today",
                over_escalated=False,
                json_valid=False,
                steps=env.step_count,
            )

    reference_urgency = result.reference_urgency or MULTITURN_TRAJECTORIES[trajectory_id].target_urgency
    predicted_urgency = result.urgency
    under = bool(result.under_escalated)
    over = (
        predicted_urgency in URGENCY_ORDER
        and reference_urgency in URGENCY_ORDER
        and URGENCY_ORDER.index(predicted_urgency) > URGENCY_ORDER.index(reference_urgency)
    )
    return EvalRow(
        trajectory_id=trajectory_id,
        policy=policy_name,
        predicted_condition=result.predicted_condition,
        predicted_urgency=predicted_urgency,
        reference_condition=result.reference_condition or MULTITURN_TRAJECTORIES[trajectory_id].target_condition,
        reference_urgency=reference_urgency,
        reward=float(result.reward),
        condition_correct=result.predicted_condition == result.reference_condition,
        urgency_correct=predicted_urgency == reference_urgency,
        under_escalated=under,
        over_escalated=over,
        json_valid=json_valid,
        steps=env.step_count,
    )


def _summarize(rows: list[EvalRow]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[EvalRow]] = {}
    for row in rows:
        by_policy.setdefault(row.policy, []).append(row)
    summary = []
    for policy, policy_rows in sorted(by_policy.items()):
        total = len(policy_rows) or 1
        summary.append(
            {
                "policy": policy,
                "cases": len(policy_rows),
                "mean_reward": round(statistics.mean(row.reward for row in policy_rows), 4),
                "condition_accuracy": round(sum(row.condition_correct for row in policy_rows) / total, 4),
                "urgency_accuracy": round(sum(row.urgency_correct for row in policy_rows) / total, 4),
                "under_escalation_rate": round(sum(row.under_escalated for row in policy_rows) / total, 4),
                "over_escalation_rate": round(sum(row.over_escalated for row in policy_rows) / total, 4),
                "json_valid_rate": round(sum(row.json_valid for row in policy_rows) / total, 4),
                "mean_steps": round(statistics.mean(row.steps for row in policy_rows), 2),
            }
        )
    return summary


def _write_markdown(path: Path, rows: list[EvalRow], summary: list[dict[str, Any]], model_notes: list[str]) -> None:
    lines = [
        "# MAAS Multi-Turn Evaluation Report",
        "",
        "This report is generated from `scripts/evaluate_multiturn.py`.",
        "It is a deterministic benchmark over the checked-in three-day trajectories.",
        "",
        "Safety note: these are synthetic benchmark metrics for a prototype triage environment, not clinical validation.",
        "",
        "## Model Evaluation Status",
        "",
        *model_notes,
        "",
        "## Summary",
        "",
        "| Policy | Cases | Mean Reward | Condition Acc. | Urgency Acc. | Under-Esc. | Over-Esc. | JSON Valid | Mean Steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['policy']} | {item['cases']} | {item['mean_reward']:.4f} | "
            f"{item['condition_accuracy']:.4f} | {item['urgency_accuracy']:.4f} | "
            f"{item['under_escalation_rate']:.4f} | {item['over_escalation_rate']:.4f} | "
            f"{item['json_valid_rate']:.4f} | {item['mean_steps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Trajectory | Policy | Prediction | Reference | Reward | Under-Esc. | Steps |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        pred = f"{row.predicted_condition}/{row.predicted_urgency}"
        ref = f"{row.reference_condition}/{row.reference_urgency}"
        lines.append(
            f"| {row.trajectory_id} | {row.policy} | {pred} | {ref} | "
            f"{row.reward:.4f} | {str(row.under_escalated).lower()} | {row.steps} |"
        )
    lines.extend(
        [
            "",
            "## Honest Interpretation",
            "",
            "- `oracle` is an upper-bound sanity check, not a deployable model.",
            "- `conservative_visible_baseline` is a simple deterministic policy using visible signals after day 3.",
            "- `day1_fast_baseline` shows how brittle one-shot triage is when it acts before evidence is revealed.",
            "- Use `--base-model`, `--trained-model`, or `--predictions-jsonl` to evaluate actual model actions under the same table before claiming trained-model improvement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MAAS multi-turn trajectory policies.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV/Markdown outputs.")
    parser.add_argument("--predictions-jsonl", help="Optional JSONL file of model actions keyed by trajectory_id.")
    parser.add_argument("--base-model", help="Optional untrained/base HF chat model id to evaluate live.")
    parser.add_argument("--trained-model", help="Optional trained GRPO HF chat model id to evaluate live.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token. Defaults to HF_TOKEN.")
    args = parser.parse_args()

    policy_builders: dict[str, Callable[[MultiTurnPrenatalEnvironment], list[Action]]] = {
        "day1_fast_baseline": day1_fast_policy,
        "conservative_visible_baseline": conservative_policy,
        "oracle": oracle_policy,
    }
    prediction_actions = _load_prediction_actions(Path(args.predictions_jsonl)) if args.predictions_jsonl else None

    rows: list[EvalRow] = []
    model_notes = [
        "- Deterministic baselines and oracle are always evaluated locally.",
        "- Base/trained model rows are included only when `--base-model` and/or `--trained-model` are supplied and Hugging Face inference succeeds.",
    ]
    if args.base_model:
        model_notes.append(f"- Base model requested: `{args.base_model}`.")
    else:
        model_notes.append("- Base model live evaluation: not run in this report.")
    if args.trained_model:
        model_notes.append(f"- Trained model requested: `{args.trained_model}`.")
    else:
        model_notes.append("- Trained model live evaluation: not run in this report.")
    for trajectory_id in sorted(MULTITURN_TRAJECTORIES):
        for policy_name, builder in policy_builders.items():
            env = MultiTurnPrenatalEnvironment()
            env.reset(trajectory_id)
            rows.append(_run_actions(trajectory_id, policy_name, builder(env)))
        if prediction_actions is not None:
            rows.append(_run_actions(trajectory_id, "model_predictions", prediction_actions.get(trajectory_id, [])))
        if args.base_model:
            rows.append(_run_hf_model(trajectory_id, f"base_model:{args.base_model}", args.base_model, args.hf_token))
        if args.trained_model:
            rows.append(_run_hf_model(trajectory_id, f"trained_model:{args.trained_model}", args.trained_model, args.hf_token))

    summary = _summarize(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "multiturn_eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    summary_path = output_dir / "multiturn_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "evaluation_report.md"
    _write_markdown(report_path, rows, summary, model_notes)

    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), "report": str(report_path), "policies": summary}, indent=2))


if __name__ == "__main__":
    main()
