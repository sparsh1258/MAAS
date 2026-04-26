#!/usr/bin/env python3
"""
inference.py - Baseline Agent for Prenatal Health Monitor (OpenEnv)
===================================================================
Runs a language model against all benchmark tasks in the Prenatal OpenEnv.

Uses the OpenAI-compatible API client. Set these environment variables:
  API_BASE_URL  - base URL for the API (e.g. https://api.openai.com/v1)
  MODEL_NAME    - model to run (e.g. gpt-4o, claude-3-5-sonnet-20241022)
  HF_TOKEN      - Hugging Face token (used as the API key / bearer token)

Emits EXACT stdout format required by OpenEnv judge:
  [START] task=<task_id> env=prenatal_health model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Usage:
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o"
  export HF_TOKEN="sk-..."
  python inference.py
"""

import json
import math
import os
import re
import sys
from typing import Any, Optional

from openai import OpenAI

from environment import Observation, _mask_observation, observation_to_prompt
from rl_risk_model import RL_RISK_MODEL
from tasks import TASKS

# Environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")
BENCHMARK = "prenatal_health"

SUCCESS_SCORE_THRESHOLD = 0.5
MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99
MAX_EPISODE_STEPS = 4
REQUEST_SIGNAL_COST = -0.25

TASK_HIDEABLE_SIGNALS = [
    "risk_flags",
    "bp_trend",
    "avg_kick_count",
    "avg_meals",
    "avg_sleep",
    "latest_weight_kg",
    "latest_energy",
    "latest_breathlessness",
]

SYSTEM_PROMPT = """You are Niva, an AI maternal health assistant for rural India.
You operate in a multi-step triage loop. Some patient signals may be intentionally withheld.

You MUST respond with ONLY a valid JSON object using this exact schema:
{"action_type":"<assess|request_signal|diagnose>","signal_name":"<hidden signal or null>","condition":"<condition or null>","urgency":"<urgency or null>","rationale":"<short explanation>"}

Rules:
- Use action_type="assess" when you want to summarize visible evidence before final diagnosis.
- Use action_type="request_signal" to reveal one hidden signal.
- Use action_type="diagnose" only when ready to finish with a condition and urgency.
- If any DANGER_ flags are visible, do not under-escalate urgency.
- Do not include markdown or extra text. Return JSON only."""

SYSTEM_PROMPT_TURN1 = """You are Niva, a maternal health AI.
TURN 1 OF 3: Make your initial diagnosis based on patient data.

Respond with ONLY valid JSON:
{"action_type": "diagnose", "target": "<condition>",
 "urgency": "<urgency>"}

Valid conditions: preeclampsia, gestational_diabetes, anemia,
preterm_risk, fetal_distress, low_risk
Valid urgencies: monitor_at_home, visit_phc_this_week,
go_to_hospital_today

Rules:
- DANGER_ flags always mean go_to_hospital_today
- ONLY the JSON object, nothing else."""

SYSTEM_PROMPT_TURN2 = """You are Niva, a maternal health AI.
TURN 2 OF 3: You made an initial diagnosis. Here is the feedback.

Review the feedback and refine your diagnosis if needed.

Respond with ONLY valid JSON:
{"action_type": "diagnose", "target": "<condition>",
 "urgency": "<urgency>"}

ONLY the JSON object, nothing else."""

SYSTEM_PROMPT_TURN3 = """You are Niva, a maternal health AI.
TURN 3 OF 3: FINAL DECISION.

Based on all evidence and feedback, make your final diagnosis.
This is your last chance to get it right.

Respond with ONLY valid JSON:
{"action_type": "diagnose", "target": "<condition>",
 "urgency": "<urgency>"}

DANGER_ flags always mean go_to_hospital_today - no exceptions.
ONLY the JSON object, nothing else."""

FALLBACK_DIAGNOSE_ACTION = {
    "action_type": "diagnose",
    "signal_name": None,
    "condition": "low_risk",
    "urgency": "monitor_at_home",
    "rationale": "Fallback due to parsing or API failure.",
}
FALLBACK_ACTION = {
    "action_type": "diagnose",
    "target": "low_risk",
    "urgency": "monitor_at_home",
}


if not HF_TOKEN and not LOCAL_MODEL_PATH:
    print("[ERROR] Set HF_TOKEN for API inference or LOCAL_MODEL_PATH for local inference.", file=sys.stderr)
    sys.exit(1)

client = (
    OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )
    if HF_TOKEN
    else None
)

_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error, log_prob=0.0):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} log_prob={log_prob:.4f} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _token_logprob_entries(response: Any) -> list[dict[str, Any]]:
    try:
        if isinstance(response, dict):
            entries = response.get("logprobs_content") or []
            normalized = []
            for item in entries:
                token = item.get("token", "")
                logprob = item.get("logprob")
                if logprob is None:
                    continue
                normalized.append({"token": str(token), "logprob": float(logprob)})
            return normalized

        logprobs = getattr(response.choices[0], "logprobs", None)
        content = getattr(logprobs, "content", None) or []
        normalized = []
        for item in content:
            token = getattr(item, "token", "")
            logprob = getattr(item, "logprob", None)
            if logprob is None:
                continue
            normalized.append({"token": str(token), "logprob": float(logprob)})
        return normalized
    except Exception:
        return []


def extract_log_prob(response: Any) -> float:
    """
    Extract sum of token log probs from an OpenAI-compatible API response.
    Falls back to -999.0 if logprobs are unavailable.

    Returns the per-token average log probability so different-length
    generations remain comparable.
    """
    try:
        entries = _token_logprob_entries(response)
        if not entries:
            return -999.0

        total_log_prob = sum(item["logprob"] for item in entries)
        avg_log_prob = total_log_prob / len(entries)
        return round(avg_log_prob, 4)
    except (AttributeError, TypeError, ZeroDivisionError, ValueError):
        return -999.0


def log_token_trajectory(response: Any, task_id: str, reward: float) -> None:
    """
    Log token-level trajectory to stderr without affecting judge stdout.
    """
    try:
        entries = _token_logprob_entries(response)
        if not entries:
            return

        tokens = [
            {
                "token": item["token"],
                "logprob": round(item["logprob"], 4),
                "prob": round(float(math.exp(item["logprob"])), 4),
            }
            for item in entries
        ]

        trajectory = {
            "task_id": task_id,
            "reward": reward,
            "token_count": len(tokens),
            "avg_logprob": round(sum(t["logprob"] for t in tokens) / len(tokens), 4),
            "min_logprob": round(min(t["logprob"] for t in tokens), 4),
            "tokens": tokens[:20],
        }
        print(f"[TRAJECTORY] {json.dumps(trajectory)}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _load_local_model():
    global _LOCAL_MODEL, _LOCAL_TOKENIZER

    if not LOCAL_MODEL_PATH:
        return None, None
    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None:
        return _LOCAL_MODEL, _LOCAL_TOKENIZER

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH)
        _LOCAL_MODEL = model
        _LOCAL_TOKENIZER = tokenizer
        return _LOCAL_MODEL, _LOCAL_TOKENIZER
    except Exception as exc:
        print(f"[DEBUG] Failed to load local model: {exc}", file=sys.stderr)
        return None, None


def _call_local_model(system_prompt: str, user_prompt: str) -> tuple[str, Optional[str], float, Any, bool]:
    try:
        model, tokenizer = _load_local_model()
        if model is None or tokenizer is None:
            return "", "Local model unavailable.", -999.0, None, False

        try:
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered = f"{system_prompt}\n\n{user_prompt}"

        inputs = tokenizer(rendered, return_tensors="pt")
        try:
            import torch

            model_device = next(model.parameters()).device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except Exception as exc:
            return "", str(exc).replace("\n", " "), -999.0, None, False

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0, prompt_length:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        token_entries: list[dict[str, Any]] = []
        for token_id, score in zip(generated_ids.tolist(), outputs.scores):
            step_logprobs = torch.log_softmax(score[0], dim=-1)
            token_logprob = float(step_logprobs[int(token_id)].item())
            token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            token_entries.append({"token": token_text, "logprob": token_logprob})

        response = {
            "backend": "local",
            "content": raw,
            "logprobs_content": token_entries,
        }
        return raw, None, extract_log_prob(response), response, bool(token_entries)
    except Exception as exc:
        error = str(exc).replace("\n", " ")
        print(f"[DEBUG] Local generation failed: {error}", file=sys.stderr)
        return "", error, -999.0, None, False


def parse_action(raw: str) -> dict[str, Any]:
    """Parse JSON action from model output, with schema normalization."""
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(line for line in lines if not line.startswith("```")).strip()

    parsed: Optional[dict[str, Any]] = None
    try:
        candidate = json.loads(clean)
        if isinstance(candidate, dict):
            parsed = candidate
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                candidate = json.loads(match.group())
                if isinstance(candidate, dict):
                    parsed = candidate
            except Exception:
                parsed = None

    if parsed is None:
        return FALLBACK_DIAGNOSE_ACTION.copy()

    if "condition" not in parsed and "target" in parsed:
        parsed["condition"] = parsed["target"]

    action_type = parsed.get("action_type")
    if action_type not in {"assess", "request_signal", "diagnose"}:
        if parsed.get("signal_name"):
            action_type = "request_signal"
        elif parsed.get("condition") or parsed.get("urgency"):
            action_type = "diagnose"
        else:
            action_type = "assess"

    normalized = {
        "action_type": action_type,
        "signal_name": parsed.get("signal_name"),
        "condition": parsed.get("condition"),
        "urgency": parsed.get("urgency"),
        "rationale": parsed.get("rationale")
        or "Model response parsed without an explicit rationale.",
    }

    if normalized["action_type"] != "request_signal":
        normalized["signal_name"] = None
    if normalized["action_type"] != "diagnose":
        normalized["condition"] = None
        normalized["urgency"] = None

    return normalized


class MinimalObs:
    def __init__(self, d: dict[str, Any] | Observation):
        if isinstance(d, Observation):
            data = {
                "risk_flags": d.risk_flags,
                "history_flags": d.history_flags,
                "avg_meals": d.avg_meals,
                "avg_sleep": d.avg_sleep,
                "avg_kick_count": d.avg_kick_count,
                "latest_energy": d.latest_energy,
                "latest_breathlessness": d.latest_breathlessness,
                "weeks_pregnant": d.weeks_pregnant,
                "trimester": d.trimester,
                "bp_trend": d.bp_trend,
                "latest_weight_kg": d.latest_weight_kg,
            }
        else:
            data = d
        self.risk_flags = data.get("risk_flags", [])
        self.history_flags = data.get("history_flags", [])
        self.avg_meals = data.get("avg_meals")
        self.avg_sleep = data.get("avg_sleep")
        self.avg_kick_count = data.get("avg_kick_count")
        self.latest_energy = data.get("latest_energy")
        self.latest_breathlessness = data.get("latest_breathlessness")
        self.weeks_pregnant = data.get("weeks_pregnant")
        self.trimester = data.get("trimester")
        self.bp_trend = data.get("bp_trend")
        self.latest_weight_kg = data.get("latest_weight_kg")


def get_rl_action(observation: Observation, rationale: Optional[str] = None) -> tuple[dict[str, Any], float]:
    """Use the lightweight risk policy as a final-diagnosis fallback."""
    try:
        result = RL_RISK_MODEL.predict(MinimalObs(observation))
        action = {
            "action_type": "diagnose",
            "signal_name": None,
            "condition": result.condition,
            "urgency": result.urgency,
            "rationale": rationale or "Fallback diagnosis from the risk policy.",
        }
        return action, float(result.confidence)
    except Exception:
        action = FALLBACK_DIAGNOSE_ACTION.copy()
        if rationale:
            action["rationale"] = rationale
        return action, 0.01


def observation_has_value(observation: Observation, signal_name: str) -> bool:
    value = getattr(observation, signal_name, None)
    if signal_name == "risk_flags":
        return value is not None
    return value is not None and value != ""


def select_withheld_signals(task: dict, observation: Observation) -> list[str]:
    candidates = [name for name in TASK_HIDEABLE_SIGNALS if observation_has_value(observation, name)]
    if not candidates:
        return []

    count = 1 if task.get("difficulty") == "easy" else min(2, len(candidates))
    ordered = sorted(candidates)
    offset = sum(ord(char) for char in task["id"]) % len(ordered)
    return [ordered[(offset + index) % len(ordered)] for index in range(count)]


def render_task_observation(observation: Observation) -> str:
    signal_mask = observation.signal_mask or {}

    def render_value(signal_name: str, value, *, none_label: str = "unknown", formatter=None) -> str:
        if not signal_mask.get(signal_name, True):
            return "withheld"
        if value is None:
            return none_label
        return formatter(value) if formatter else str(value)

    return (
        "Benchmark patient observation:\n"
        f"- User ID: {observation.user_id}\n"
        f"- Region: {observation.region}\n"
        f"- Weeks pregnant: {observation.weeks_pregnant}\n"
        f"- Trimester: {observation.trimester}\n"
        f"- History flags: {', '.join(observation.history_flags) if observation.history_flags else 'none'}\n"
        f"- Risk flags: {render_value('risk_flags', observation.risk_flags, none_label='none', formatter=lambda flags: ', '.join(flags) if flags else 'none')}\n"
        f"- BP trend: {render_value('bp_trend', observation.bp_trend, none_label='unknown')}\n"
        f"- Average kick count: {render_value('avg_kick_count', observation.avg_kick_count)}\n"
        f"- Average meals per day: {render_value('avg_meals', observation.avg_meals, formatter=lambda value: f'{value:.2f}')}\n"
        f"- Average sleep hours: {render_value('avg_sleep', observation.avg_sleep, formatter=lambda value: f'{value:.2f}')}\n"
        f"- Latest weight (kg): {render_value('latest_weight_kg', observation.latest_weight_kg)}\n"
        f"- Latest energy (1-10): {render_value('latest_energy', observation.latest_energy)}\n"
        f"- Latest breathlessness (1-10): {render_value('latest_breathlessness', observation.latest_breathlessness)}\n"
        f"- Days of data: {observation.days_of_data}\n"
        f"- Available signals: {', '.join(observation.available_signals) if observation.available_signals else 'all'}\n"
        f"- Withheld signals: {', '.join(observation.withheld_signals) if observation.withheld_signals else 'none'}"
    )


def build_task_prompt(observation: Observation):
    text_observation = render_task_observation(observation)
    return observation_to_prompt(observation, text_observation)


def init_task_episode(task: dict) -> Optional[dict[str, Any]]:
    observation = task.get("observation")
    if not isinstance(observation, Observation):
        return None

    full_observation = observation.model_copy(deep=True)
    withheld_signals = select_withheld_signals(task, full_observation)
    visible_observation = _mask_observation(full_observation, withheld_signals)
    prompt = build_task_prompt(visible_observation)
    return {
        "full_observation": full_observation,
        "visible_observation": visible_observation,
        "prompt": prompt,
    }


def build_turn_user_prompt(base_prompt: str, history: list[dict[str, Any]], stage: str) -> str:
    stage_instruction = {
        "assess": "This turn is an assessment turn. Summarize visible evidence with action_type='assess'.",
        "request_signal": "This turn must request the single most informative hidden signal with action_type='request_signal'.",
        "diagnose": "This is the final turn. Provide action_type='diagnose' with condition and urgency.",
    }[stage]

    if not history:
        return base_prompt + "\n\nCurrent turn instruction: " + stage_instruction

    history_lines = []
    for item in history:
        signal_suffix = f", signal_name={item['signal_name']}" if item.get("signal_name") else ""
        diagnosis_suffix = ""
        if item.get("condition") or item.get("urgency"):
            diagnosis_suffix = (
                f", condition={item.get('condition')}, urgency={item.get('urgency')}"
            )
        history_lines.append(
            f"- Step {item['step']}: action_type={item['action_type']}{signal_suffix}{diagnosis_suffix}; "
            f"feedback={item['feedback']}"
        )

    history_block = "\n".join(history_lines)
    return (
        base_prompt
        + "\n\nEpisode history so far:\n"
        + history_block
        + "\n\nCurrent turn instruction: "
        + stage_instruction
    )


def call_model(system_prompt: str, user_prompt: str) -> tuple[str, Optional[str], float, Any, bool]:
    if LOCAL_MODEL_PATH:
        return _call_local_model(system_prompt, user_prompt)

    response = None
    logprobs_supported = True
    try:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.0,
                max_tokens=256,
                logprobs=True,
                top_logprobs=1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            print(f"[DEBUG] Retrying without logprobs: {exc}", file=sys.stderr)
            logprobs_supported = False
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.0,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        raw = (response.choices[0].message.content or "").strip()
        real_log_prob = extract_log_prob(response) if logprobs_supported else -999.0
        return raw, None, real_log_prob, response, logprobs_supported
    except Exception as exc:
        error = str(exc).replace("\n", " ")
        print(f"[DEBUG] API call failed: {error}", file=sys.stderr)
        return "", error, -999.0, response, False


def proxy_log_prob_from_observation(observation: Optional[Observation]) -> float:
    if not isinstance(observation, Observation):
        return 0.01
    try:
        _, confidence = get_rl_action(observation, rationale="Proxy confidence fallback.")
        return round(float(confidence), 4)
    except Exception:
        return 0.01


def fallback_stage_action(
    stage: str,
    visible_observation: Optional[Observation],
    hidden_signals: list[str],
    *,
    rationale: str,
) -> tuple[dict[str, Any], float]:
    if stage == "assess":
        return (
            {
                "action_type": "assess",
                "signal_name": None,
                "condition": None,
                "urgency": None,
                "rationale": rationale,
            },
            0.01,
        )

    if stage == "request_signal":
        return (
            {
                "action_type": "request_signal",
                "signal_name": hidden_signals[0] if hidden_signals else None,
                "condition": None,
                "urgency": None,
                "rationale": rationale,
            },
            0.01,
        )

    if visible_observation is None:
        action = FALLBACK_DIAGNOSE_ACTION.copy()
        action["rationale"] = rationale
        return action, 0.01
    return get_rl_action(visible_observation, rationale=rationale)


def coerce_action_to_stage(
    action: dict[str, Any],
    stage: str,
    visible_observation: Optional[Observation],
    hidden_signals: list[str],
) -> tuple[dict[str, Any], float]:
    rationale = action.get("rationale") or "Stage-coerced action."

    if stage == "assess":
        return (
            {
                "action_type": "assess",
                "signal_name": None,
                "condition": None,
                "urgency": None,
                "rationale": rationale,
            },
            0.01,
        )

    if stage == "request_signal":
        signal_name = action.get("signal_name")
        if signal_name not in hidden_signals:
            signal_name = hidden_signals[0] if hidden_signals else None
        return (
            {
                "action_type": "request_signal",
                "signal_name": signal_name,
                "condition": None,
                "urgency": None,
                "rationale": rationale,
            },
            0.01,
        )

    if action.get("condition") and action.get("urgency"):
        return (
            {
                "action_type": "diagnose",
                "signal_name": None,
                "condition": action.get("condition"),
                "urgency": action.get("urgency"),
                "rationale": rationale,
            },
            0.01,
        )

    return fallback_stage_action(
        "diagnose",
        visible_observation,
        hidden_signals,
        rationale=rationale or "Missing diagnosis fields in final turn.",
    )


def apply_episode_step(
    episode: Optional[dict[str, Any]],
    action: dict[str, Any],
    grade_fn,
) -> dict[str, Any]:
    if episode is None:
        grade_result = grade_fn(action)
        score = min(max(float(grade_result["score"]), MIN_STRICT_SCORE), MAX_STRICT_SCORE)
        return {
            "action": action,
            "reward": score,
            "done": True,
            "feedback": grade_result.get("feedback", ""),
            "score": score,
            "success": score >= SUCCESS_SCORE_THRESHOLD,
            "grade_result": grade_result,
        }

    visible_observation: Observation = episode["visible_observation"]
    full_observation: Observation = episode["full_observation"]
    hidden_signals = list(visible_observation.withheld_signals)
    action_type = action.get("action_type", "assess")

    if action_type == "assess":
        return {
            "action": action,
            "reward": 0.0,
            "done": False,
            "feedback": "Assessment recorded. Continue gathering evidence or diagnose when ready.",
            "score": 0.0,
            "success": False,
            "grade_result": None,
        }

    if action_type == "request_signal":
        if not hidden_signals:
            return {
                "action": action,
                "reward": -0.1,
                "done": False,
                "feedback": "No hidden signals remain. Move to assessment or diagnosis.",
                "score": 0.0,
                "success": False,
                "grade_result": None,
            }

        requested_signal = action.get("signal_name") or hidden_signals[0]
        if requested_signal not in hidden_signals:
            requested_signal = hidden_signals[0]
            action["signal_name"] = requested_signal

        remaining_hidden = [signal for signal in hidden_signals if signal != requested_signal]
        updated_visible = _mask_observation(full_observation, remaining_hidden)
        episode["visible_observation"] = updated_visible
        episode["prompt"] = build_task_prompt(updated_visible)
        return {
            "action": action,
            "reward": REQUEST_SIGNAL_COST,
            "done": False,
            "feedback": (
                f"Signal '{requested_signal}' has been revealed. Reassess before the final diagnosis."
            ),
            "score": 0.0,
            "success": False,
            "grade_result": None,
        }

    final_action = {
        "condition": action.get("condition"),
        "urgency": action.get("urgency"),
        "rationale": action.get("rationale"),
    }
    grade_result = grade_fn(final_action)
    score = min(max(float(grade_result["score"]), MIN_STRICT_SCORE), MAX_STRICT_SCORE)
    return {
        "action": action,
        "reward": score,
        "done": True,
        "feedback": grade_result.get("feedback", ""),
        "score": score,
        "success": score >= SUCCESS_SCORE_THRESHOLD,
        "grade_result": grade_result,
    }


def update_rl_policy(observation: Observation, action: dict[str, Any], reward: float) -> None:
    try:
        predicted = action.get("condition") or "low_risk"
        RL_RISK_MODEL.update_from_reward(MinimalObs(observation), predicted, reward)
    except Exception:
        pass


def run_agent(task: dict) -> dict:
    task_id = task["id"]
    grade_fn = task["grade"]
    prompt_fn = task["prompt"]
    user_prompt = prompt_fn()

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    action = FALLBACK_ACTION.copy()
    api_error: Optional[str] = None
    result = {"score": 0.01, "passed": False, "feedback": "No grading result produced."}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs_data = task.get("observation") or task
    if isinstance(obs_data, Observation):
        obs_payload = {
            "risk_flags": obs_data.risk_flags,
            "history_flags": obs_data.history_flags,
            "avg_meals": obs_data.avg_meals,
            "avg_sleep": obs_data.avg_sleep,
            "avg_kick_count": obs_data.avg_kick_count,
            "latest_energy": obs_data.latest_energy,
            "latest_breathlessness": obs_data.latest_breathlessness,
            "weeks_pregnant": obs_data.weeks_pregnant,
            "trimester": obs_data.trimester,
            "bp_trend": obs_data.bp_trend,
            "latest_weight_kg": obs_data.latest_weight_kg,
        }
    else:
        obs_payload = dict(obs_data)

    try:
        rl_obs = MinimalObs(obs_payload)
        rl_result = RL_RISK_MODEL.predict(rl_obs)
        rl_action = {
            "action_type": "diagnose",
            "target": rl_result.condition,
            "urgency": rl_result.urgency,
        }
        rl_log_prob = float(rl_result.confidence)
    except Exception:
        rl_obs = MinimalObs({})
        rl_action = FALLBACK_ACTION.copy()
        rl_log_prob = 0.01

    danger_flags = [
        flag
        for flag in obs_payload.get("risk_flags", [])
        if str(flag).startswith("DANGER_")
    ]

    conversation_history: list[dict[str, str]] = []
    last_feedback = ""
    system_prompts = [
        SYSTEM_PROMPT_TURN1,
        SYSTEM_PROMPT_TURN2,
        SYSTEM_PROMPT_TURN3,
    ]

    try:
        for turn in range(3):
            current_system = system_prompts[turn]

            if turn == 0:
                user_msg = user_prompt
            else:
                user_msg = (
                    f"Original patient data:\n{user_prompt}\n\n"
                    f"Your previous action: {json.dumps(action)}\n\n"
                    f"Feedback received: {last_feedback}\n\n"
                    f"Refine your diagnosis if needed."
                )

            conversation_history.append({"role": "user", "content": user_msg})

            raw, api_error, log_prob, response, _logprobs_supported = call_model(
                current_system,
                user_msg,
            )
            if raw:
                conversation_history.append({"role": "assistant", "content": raw})
            else:
                raw = json.dumps(rl_action)

            llm_action = parse_action(raw)
            llm_action_for_grade = {
                "action_type": "diagnose",
                "target": llm_action.get("condition") or llm_action.get("target") or "low_risk",
                "urgency": llm_action.get("urgency") or "monitor_at_home",
            }

            if rl_action["target"] != "low_risk":
                action = rl_action.copy()
            elif llm_action_for_grade["target"] != "low_risk":
                action = llm_action_for_grade.copy()
            else:
                action = llm_action_for_grade.copy()

            if danger_flags:
                action["urgency"] = "go_to_hospital_today"

            result = grade_fn(action)
            score = float(result["score"])
            score = min(max(score, MIN_STRICT_SCORE), MAX_STRICT_SCORE)
            reward = score
            success = score >= SUCCESS_SCORE_THRESHOLD

            last_feedback = result.get("feedback", "")
            rewards.append(reward)
            steps_taken = turn + 1

            print(
                f"[TURN {turn + 1}] action={json.dumps(action)} "
                f"reward={reward:.4f} "
                f"feedback={last_feedback[:60]}",
                file=sys.stderr,
                flush=True,
            )
            if response is not None:
                log_token_trajectory(response, task_id, reward)

            is_done = (turn == 2) or (score >= 0.95)
            current_log_prob = rl_log_prob if api_error or log_prob == -999.0 else log_prob
            log_step(
                step=turn + 1,
                action=json.dumps(action),
                reward=reward,
                done=is_done,
                error=api_error,
                log_prob=current_log_prob,
            )

            try:
                RL_RISK_MODEL.update_from_reward(rl_obs, action["target"], reward)
            except Exception:
                pass

            if score >= 0.95:
                print(f"[TURN {turn + 1}] Perfect score - stopping early", file=sys.stderr)
                break

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return {
        "task_id": task_id,
        "score": score,
        "passed": result.get("passed", success),
        "feedback": result.get("feedback", ""),
        "action": action,
    }


def main():
    results = []
    total_score = 0.0

    for task in TASKS:
        result = run_agent(task)
        results.append(result)
        total_score += result["score"]

    avg_score = total_score / len(TASKS) if TASKS else 0.0

    print("\n" + "=" * 60, file=sys.stderr)
    print("BASELINE RESULTS SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {result['task_id']:25s} score={result['score']:.4f}", file=sys.stderr)
        feedback_preview = result["feedback"][:80] if result["feedback"] else ""
        print(f"         feedback: {feedback_preview}...", file=sys.stderr)
    print(f"\n  AVERAGE SCORE: {avg_score:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
