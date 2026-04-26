from __future__ import annotations

from typing import Any

from environment import MULTITURN_TRAJECTORIES, MultiTurnPrenatalEnvironment

TASK_ID = "task_4_multiturn_easy"
TASK_NAME = "Multi-Turn Preeclampsia Escalation"
TASK_DESCRIPTION = (
    "Track a slow-rising preeclampsia case across three days, gather the later-day evidence, "
    "and escalate to hospital with the correct final diagnosis."
)
TASK_TRAJECTORY_ID = "traj_preeclampsia_slow"
TASK_TRAJECTORY = MULTITURN_TRAJECTORIES[TASK_TRAJECTORY_ID]

EXPECTED_CONDITION = TASK_TRAJECTORY.target_condition
EXPECTED_URGENCY = TASK_TRAJECTORY.target_urgency
EXPECTED_ACTION_SEQUENCE = [
    {"action_type": "advance_day"},
    {"action_type": "advance_day"},
    {"action_type": "diagnose", "target": EXPECTED_CONDITION, "urgency": EXPECTED_URGENCY},
]


def create_environment() -> MultiTurnPrenatalEnvironment:
    env = MultiTurnPrenatalEnvironment()
    env.reset(TASK_TRAJECTORY_ID)
    return env


def teacher_actions() -> list[dict[str, Any]]:
    return [dict(action) for action in EXPECTED_ACTION_SEQUENCE]


def grade(action: dict[str, Any]) -> dict[str, Any]:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")
    trace = action.get("episode_trace") or action.get("trace") or []
    action_types = [step.get("action", {}).get("action_type") for step in trace if isinstance(step, dict)]

    score = 0.01
    feedback = (
        "The episode should gather day-2 and day-3 evidence before issuing the final diagnosis."
    )

    if action_types[:2] == ["advance_day", "advance_day"]:
        score = 0.4
        feedback = "Good information gathering: you advanced through both early days before diagnosing."

    if predicted == EXPECTED_CONDITION:
        score = max(score, 0.7)
        feedback = "Correct condition selected for the slow-rising hypertensive trajectory."

    if predicted == EXPECTED_CONDITION and urgency == EXPECTED_URGENCY:
        score = 0.99
        feedback = (
            "Perfect: the agent waited for the later evidence, recognized preeclampsia, "
            "and escalated to hospital appropriately."
        )
    elif predicted == EXPECTED_CONDITION and urgency == "visit_phc_this_week":
        score = max(score, 0.55)
        feedback = "Condition is correct, but PHC referral is too slow once the day-3 danger signs appear."
    elif predicted == "low_risk":
        score = 0.01
        feedback = "Incorrect: rising blood pressure with later neurologic symptoms should not be labeled low risk."

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
        },
    }


def get_task_prompt() -> str:
    return (
        "Use MultiTurnPrenatalEnvironment on trajectory "
        f"`{TASK_TRAJECTORY_ID}`. The agent should advance through day 2 and day 3, "
        "then submit the final diagnosis once the full picture is visible."
    )
