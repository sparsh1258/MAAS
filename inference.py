#!/usr/bin/env python3
"""
inference.py — Baseline Agent for Prenatal Health Monitor (OpenEnv)
=====================================================================
Runs a language model against all 3 tasks in the Prenatal OpenEnv.

Uses the OpenAI-compatible API client. Set these environment variables:
  API_BASE_URL  — base URL for the API (e.g. https://api.openai.com/v1)
  MODEL_NAME    — model to run (e.g. gpt-4o, claude-3-5-sonnet-20241022)
  HF_TOKEN      — Hugging Face token (used as the API key / bearer token)

Emits EXACT stdout format required by OpenEnv judge:
  [START] task=<task_id> env=prenatal_health model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Any deviation = disqualified.

Usage:
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o"
  export HF_TOKEN="sk-..."
  python inference.py
"""

import os
import sys
import json
import re
from typing import Optional
from openai import OpenAI
from tasks import TASKS

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN")
BENCHMARK    = "prenatal_health"

SUCCESS_SCORE_THRESHOLD = 0.5  # score >= 0.5 counts as success
MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

SYSTEM_PROMPT = """You are Niva, an AI maternal health assistant for rural India.
You will be given a patient observation and must diagnose the primary condition.

You MUST respond with ONLY a valid JSON object in this exact format:
{"action_type": "diagnose", "target": "<condition>", "urgency": "<urgency>"}

Valid conditions: preeclampsia, gestational_diabetes, anemia, preterm_risk, fetal_distress, low_risk
Valid urgencies:  monitor_at_home, visit_phc_this_week, go_to_hospital_today

Rules:
- If any DANGER_ flags are present in risk_flags, urgency MUST be go_to_hospital_today
- DANGER_BP_CRITICAL means BP >= 160/110 — always go_to_hospital_today
- DANGER_LOW_KICKS means kick count < 3 — always go_to_hospital_today
- DANGER_BLEEDING means active bleeding — always go_to_hospital_today
- HIGH_BP (without DANGER) means >= 140/90 — consider visit_phc_this_week
- If no risk flags and no history: likely low_risk with monitor_at_home
- Do NOT include any explanation, markdown, or extra text — ONLY the JSON object."""


# ── Stdout logging helpers (spec-compliant) ────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Action parsing ─────────────────────────────────────────────────────────────

FALLBACK_ACTION = {"action_type": "diagnose", "target": "low_risk", "urgency": "monitor_at_home"}


def parse_action(raw: str) -> dict:
    """Parse JSON action from model output, with fallback."""
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(l for l in lines if not l.startswith("```")).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return FALLBACK_ACTION.copy()


# ── Single task runner ─────────────────────────────────────────────────────────

def run_agent(task: dict) -> dict:
    """
    Run the agent on a single task.
    Emits one [START], one [STEP] (single-step task), one [END].
    Returns grading result dict.
    """
    task_id  = task["id"]
    grade_fn = task["grade"]
    prompt_fn = task["prompt"]

    user_prompt = prompt_fn()

    rewards    = []
    steps_taken = 0
    score      = 0.0
    success    = False
    action     = FALLBACK_ACTION.copy()
    api_error  = None
    result     = {"score": 0.01, "passed": False, "feedback": "No grading result produced."}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Call the model ─────────────────────────────────────────────────────
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.0,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            api_error = None
        except Exception as e:
            raw = json.dumps(FALLBACK_ACTION)
            api_error = str(e).replace("\n", " ")
            print(f"[DEBUG] API call failed for {task_id}: {e}", file=sys.stderr)

        # ── Parse action ───────────────────────────────────────────────────────
        action = parse_action(raw)

        # ── Grade ──────────────────────────────────────────────────────────────
        result  = grade_fn(action)
        score   = float(result["score"])
        score   = min(max(score, MIN_STRICT_SCORE), MAX_STRICT_SCORE)   # clamp to strict (0, 1)
        reward  = score                        # single-step: reward == score
        success = score >= SUCCESS_SCORE_THRESHOLD

        rewards.append(reward)
        steps_taken = 1

        # ── [STEP] ─────────────────────────────────────────────────────────────
        log_step(
            step=1,
            action=json.dumps(action),
            reward=reward,
            done=True,
            error=api_error,
        )

    finally:
        # ── [END] always emitted, even on unexpected exception ─────────────────
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":  task_id,
        "score":    score,
        "passed":   result.get("passed", success),
        "feedback": result.get("feedback", ""),
        "action":   action,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    results    = []
    total_score = 0.0

    for task in TASKS:
        result = run_agent(task)
        results.append(result)
        total_score += result["score"]

    avg_score = total_score / len(TASKS) if TASKS else 0.0

    # ── Final summary to stderr (not captured by judge) ───────────────────────
    print("\n" + "=" * 60, file=sys.stderr)
    print("BASELINE RESULTS SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['task_id']:25s} score={r['score']:.4f}", file=sys.stderr)
        feedback_preview = r["feedback"][:80] if r["feedback"] else ""
        print(f"         feedback: {feedback_preview}...", file=sys.stderr)
    print(f"\n  AVERAGE SCORE: {avg_score:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
