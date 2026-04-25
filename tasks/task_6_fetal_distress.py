"""
Task 6 - HARD: Fetal Distress Emergency
=======================================
Objective:
  Identify fetal_distress when low kicks are the dominant danger signal and
  escalate immediately.
"""

from environment import Observation

TASK_ID = "task_6_fetal_distress"
TASK_NAME = "Fetal Distress Emergency"
TASK_DESCRIPTION = (
    "Handle a low-kick emergency where fetal_distress is the primary diagnosis and hospital escalation is mandatory."
)

TASK_OBSERVATION = Observation(
    user_id=9006,
    weeks_pregnant=36,
    trimester=3,
    region="Odisha",
    risk_flags=["DANGER_LOW_KICKS", "LOW_KICK_AVG", "DIZZINESS_SIGNAL"],
    bp_trend="stable",
    avg_kick_count=2.4,
    avg_meals=2.8,
    avg_sleep=5.6,
    latest_weight_kg=63.0,
    latest_energy=4,
    latest_breathlessness=5,
    history_flags=[],
    days_of_data=3,
)

EXPECTED_CONDITION = "fetal_distress"
EXPECTED_URGENCY = "go_to_hospital_today"


def grade(action: dict) -> dict:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    if predicted == EXPECTED_CONDITION and urgency == EXPECTED_URGENCY:
        score = 0.99
        feedback = "Perfect: fetal_distress correctly identified and urgently escalated."
    elif predicted == "preterm_risk" and urgency == EXPECTED_URGENCY:
        score = 0.45
        feedback = "Urgency is correct, but preterm_risk misses that low kicks are the dominant emergency signal."
    elif urgency == "visit_phc_this_week":
        score = 0.15
        feedback = "Under-escalated: DANGER_LOW_KICKS should force go_to_hospital_today."
    elif urgency == "monitor_at_home":
        score = 0.01
        feedback = "Critical failure: monitor_at_home is unsafe with DANGER_LOW_KICKS."
    elif urgency == EXPECTED_URGENCY:
        score = 0.3
        feedback = f"Urgency was correct, but condition '{predicted}' was not."
    else:
        score = 0.01
        feedback = (
            f"Incorrect diagnosis ('{predicted}') and urgency ('{urgency}'). Expected fetal_distress / go_to_hospital_today."
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
Risk Flags       : {obs.risk_flags}
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
