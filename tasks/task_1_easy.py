"""
Task 1 — EASY: Low-Risk Patient Assessment
==========================================
Objective:
  Given a healthy 2nd-trimester patient with normal BP, good kick counts,
  no danger flags, and no medical history — correctly identify her as low_risk
  with urgency monitor_at_home.

Difficulty: Easy
  - Clear, unambiguous signals all pointing to low_risk
  - No competing conditions
  - No history flags

Expected output:
  condition  = low_risk
  urgency    = monitor_at_home
  score      = 1.0 (full credit for exact match)

Partial credit:
  - Correct condition, wrong urgency (over-escalated): 0.5
  - Wrong condition, correct urgency: 0.2
  - Both wrong: 0.0
"""

from environment import Observation, ActionModel, PrenatalEnvironment

TASK_ID = "task_1_easy"
TASK_NAME = "Low-Risk Patient Assessment"
TASK_DESCRIPTION = (
    "Assess a healthy 2nd-trimester patient with normal vitals and no history. "
    "Correctly classify as low_risk with monitor_at_home urgency."
)

# Deterministic synthetic observation — no DB dependency for grading
TASK_OBSERVATION = Observation(
    user_id=9001,
    weeks_pregnant=20,
    trimester=2,
    region="Rajasthan",
    risk_flags=[],
    bp_trend="stable",
    avg_kick_count=10.0,
    avg_meals=3.0,
    avg_sleep=7.5,
    latest_weight_kg=62.0,
    latest_energy=7,
    latest_breathlessness=2,
    history_flags=[],
    days_of_data=3,
)

EXPECTED_CONDITION = "low_risk"
EXPECTED_URGENCY = "monitor_at_home"


def grade(action: dict) -> dict:
    """
    Grade the agent's action against this task.

    Args:
        action: dict with keys condition, urgency, rationale

    Returns:
        dict with keys: score (0.0-1.0), passed (bool), feedback (str)
    """
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    correct_condition = predicted == EXPECTED_CONDITION
    correct_urgency = urgency == EXPECTED_URGENCY

    if correct_condition and correct_urgency:
        score = 0.99
        feedback = "Perfect: correctly identified low_risk with monitor_at_home urgency."
    elif correct_condition and urgency == "visit_phc_this_week":
        score = 0.5
        feedback = (
            f"Condition correct (low_risk) but urgency '{urgency}' is over-escalated. "
            "A healthy patient doesn't need a PHC visit this week."
        )
    elif correct_condition and urgency == "go_to_hospital_today":
        score = 0.3
        feedback = (
            "Condition correct (low_risk) but 'go_to_hospital_today' is a significant "
            "over-escalation for a patient with no risk flags."
        )
    elif not correct_condition and correct_urgency:
        score = 0.2
        feedback = (
            f"Urgency correct (monitor_at_home) but condition '{predicted}' is wrong. "
            "There are no signals for any specific condition here."
        )
    else:
        score = 0.01
        feedback = (
            f"Both condition ('{predicted}') and urgency ('{urgency}') are incorrect. "
            f"Expected: condition={EXPECTED_CONDITION}, urgency={EXPECTED_URGENCY}. "
            "Review the observation: no risk flags, normal BP, good kicks, no history."
        )

    return {
        "score": round(score, 4),
        "passed": score >= 0.99,
        "feedback": feedback,
        "expected": {"condition": EXPECTED_CONDITION, "urgency": EXPECTED_URGENCY},
        "predicted": {"condition": predicted, "urgency": urgency},
    }


def get_task_prompt() -> str:
    """Return the natural-language prompt for the agent."""
    obs = TASK_OBSERVATION
    return f"""
You are Niva, an AI maternal health assistant working in rural India.

A pregnant woman's health data is shown below. Based ONLY on this data,
diagnose her condition and set the appropriate urgency level.

=== PATIENT OBSERVATION ===
Weeks Pregnant   : {obs.weeks_pregnant} (Trimester {obs.trimester})
Region           : {obs.region}
Risk Flags       : {obs.risk_flags or 'None'}
BP Trend         : {obs.bp_trend}
Avg Kick Count   : {obs.avg_kick_count}
Avg Meals/Day    : {obs.avg_meals}
Avg Sleep Hours  : {obs.avg_sleep}
Latest Weight    : {obs.latest_weight_kg} kg
Energy Level     : {obs.latest_energy}/10
Breathlessness   : {obs.latest_breathlessness}/10
Medical History  : {obs.history_flags or 'None'}
Days of Data     : {obs.days_of_data}

=== YOUR TASK ===
Call the diagnose action with:
  - condition: one of [preeclampsia, gestational_diabetes, anemia, preterm_risk, fetal_distress, low_risk]
  - urgency: one of [monitor_at_home, visit_phc_this_week, go_to_hospital_today]
  - rationale: a short clinical explanation

Respond in JSON:
{{"condition": "<condition>", "urgency": "<urgency>", "rationale": "<short explanation>"}}
""".strip()
