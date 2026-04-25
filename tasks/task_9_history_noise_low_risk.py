"""
Task 9 - MEDIUM: Reassuring Case With History Noise
===================================================
Objective:
  Avoid over-escalation when the patient has some family history but the active
  observation is otherwise reassuring and low-risk.
"""

from environment import Observation

TASK_ID = "task_9_history_noise_low_risk"
TASK_NAME = "Reassuring Case With History Noise"
TASK_DESCRIPTION = (
    "Recognize a reassuring low-risk case even when family history might tempt unnecessary escalation."
)

TASK_OBSERVATION = Observation(
    user_id=9009,
    weeks_pregnant=29,
    trimester=3,
    region="Maharashtra",
    risk_flags=[],
    bp_trend="stable",
    avg_kick_count=9.2,
    avg_meals=3.3,
    avg_sleep=7.1,
    latest_weight_kg=66.0,
    latest_energy=7,
    latest_breathlessness=2,
    history_flags=["family_diabetes"],
    days_of_data=3,
)

EXPECTED_CONDITION = "low_risk"
EXPECTED_URGENCY = "monitor_at_home"


def grade(action: dict) -> dict:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    if predicted == EXPECTED_CONDITION and urgency == EXPECTED_URGENCY:
        score = 0.99
        feedback = "Perfect: correctly stayed low-risk despite non-acute family history."
    elif predicted == EXPECTED_CONDITION and urgency == "visit_phc_this_week":
        score = 0.55
        feedback = "Condition is right, but the case is reassuring enough for monitor_at_home."
    elif predicted == "gestational_diabetes" and urgency == "visit_phc_this_week":
        score = 0.25
        feedback = "Family history alone is not enough to make gestational_diabetes the primary diagnosis here."
    elif urgency == EXPECTED_URGENCY:
        score = 0.2
        feedback = f"Urgency was correct, but condition '{predicted}' was not."
    else:
        score = 0.01
        feedback = (
            f"Incorrect diagnosis ('{predicted}') and urgency ('{urgency}'). "
            "Expected low_risk / monitor_at_home."
        )

    return {
        "score": round(score, 4),
        "passed": score >= 0.99,
        "feedback": feedback,
        "expected": {"condition": EXPECTED_CONDITION, "urgency": EXPECTED_URGENCY},
        "predicted": {"condition": predicted, "urgency": urgency},
    }


def get_task_prompt() -> str:
    obs = TASK_OBSERVATION
    return f"""
You are Niva, an AI maternal health assistant working in rural India.

Assess the patient below and decide the primary maternal-health condition and urgency.

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
Medical History  : {obs.history_flags}
Days of Data     : {obs.days_of_data}

=== YOUR TASK ===
Call the diagnose action with:
  - condition: one of [preeclampsia, gestational_diabetes, anemia, preterm_risk, fetal_distress, low_risk]
  - urgency: one of [monitor_at_home, visit_phc_this_week, go_to_hospital_today]
  - rationale: a short clinical explanation

Respond in JSON:
{{"condition": "<condition>", "urgency": "<urgency>", "rationale": "<short explanation>"}}
""".strip()
