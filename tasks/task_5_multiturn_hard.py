from __future__ import annotations

from typing import Any

from environment import MULTITURN_TRAJECTORIES, MultiTurnPrenatalEnvironment

TASK_ID = "task_5_multiturn_hard"
TASK_NAME = "Multi-Turn Mixed-Signals Reasoning"
TASK_DESCRIPTION = (
    "Handle a genuinely ambiguous trajectory with hypertension, low kicks, and diabetes history "
    "by using request actions before making the final diagnosis."
)
TASK_TRAJECTORY_ID = "traj_mixed_signals_hard"
TASK_TRAJECTORY = MULTITURN_TRAJECTORIES[TASK_TRAJECTORY_ID]

EXPECTED_CONDITION = TASK_TRAJECTORY.target_condition
EXPECTED_URGENCY = TASK_TRAJECTORY.target_urgency


def create_environment() -> MultiTurnPrenatalEnvironment:
    env = MultiTurnPrenatalEnvironment()
    env.reset(TASK_TRAJECTORY_ID)
    return env


def teacher_actions() -> list[dict[str, Any]]:
    return [
        {"action_type": "request_bp_recheck"},
        {"action_type": "advance_day"},
        {"action_type": "request_kick_count"},
        {"action_type": "advance_day"},
        {"action_type": "diagnose", "target": EXPECTED_CONDITION, "urgency": EXPECTED_URGENCY},
    ]


def grade(action: dict[str, Any]) -> dict[str, Any]:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")
    trace = action.get("episode_trace") or action.get("trace") or []
    action_types = [step.get("action", {}).get("action_type") for step in trace if isinstance(step, dict)]

    used_request_action = any(name in {"request_bp_recheck", "request_kick_count"} for name in action_types)
    diagnosed_too_early = "diagnose" in action_types[:2]

    score = 0.01
    feedback = "This task expects deliberate information gathering before the final diagnosis."

    if used_request_action:
        score = 0.35
        feedback = "Good: the agent used at least one request action to reduce uncertainty."

    if used_request_action and predicted == EXPECTED_CONDITION:
        score = 0.7
        feedback = "Strong reasoning: the agent gathered evidence and reached the right primary condition."

    if predicted == EXPECTED_CONDITION and urgency == EXPECTED_URGENCY and used_request_action and not diagnosed_too_early:
        score = 0.99
        feedback = (
            "Perfect: the agent gathered extra evidence, avoided a premature call, "
            "and escalated the mixed-signal case correctly."
        )
    elif predicted == EXPECTED_CONDITION and urgency != EXPECTED_URGENCY:
        score = max(score, 0.6)
        feedback = "Condition is correct, but urgency is still too low for the combined danger pattern."
    elif predicted == "gestational_diabetes":
        score = 0.2
        feedback = "Family diabetes history is a distractor here; the later-day safety pattern points elsewhere."
    elif diagnosed_too_early:
        score = 0.05
        feedback = "Premature diagnosis: this hard task is designed to reward information gathering first."

    return {
        "score": round(score, 4),
        "passed": score >= 0.99,
        "feedback": feedback,
        "expected": {
            "trajectory_id": TASK_TRAJECTORY_ID,
            "condition": EXPECTED_CONDITION,
            "urgency": EXPECTED_URGENCY,
            "teacher_actions": teacher_actions(),
        },
        "predicted": {
            "condition": predicted,
            "urgency": urgency,
            "observed_action_types": action_types,
            "used_request_action": used_request_action,
        },
    }


def get_task_prompt() -> str:
    return (
        "Use MultiTurnPrenatalEnvironment on trajectory "
        f"`{TASK_TRAJECTORY_ID}`. The agent should request more evidence before committing to a diagnosis."
    )
