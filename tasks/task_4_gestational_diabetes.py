"""
Task 4 - MEDIUM: Gestational Diabetes Screening
================================================
Objective:
  Identify gestational_diabetes in a later-pregnancy patient with diabetes
  family history, unusually high meals, low energy, and breathlessness,
  without immediate danger flags.

Expected output:
  condition = gestational_diabetes
  urgency   = visit_phc_this_week
"""

from environment import Observation

TASK_ID = "task_4_gestational_diabetes"
TASK_NAME = "Gestational Diabetes Screening"
TASK_DESCRIPTION = (
    "Detect gestational_diabetes from later-pregnancy metabolic strain signals "
    "and family diabetes history without immediate danger flags."
)

TASK_OBSERVATION = Observation(
    user_id=9004,
    weeks_pregnant=31,
    trimester=3,
    region="Madhya Pradesh",
    risk_flags=[],
    bp_trend="stable",
    avg_kick_count=8.2,
    avg_meals=4.2,
    avg_sleep=6.2,
    latest_weight_kg=79.0,
    latest_energy=4,
    latest_breathlessness=7,
    history_flags=["family_diabetes"],
    days_of_data=3,
)

EXPECTED_CONDITION = "gestational_diabetes"
EXPECTED_URGENCY = "visit_phc_this_week"


def grade(action: dict) -> dict:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    if predicted == EXPECTED_CONDITION and urgency == EXPECTED_URGENCY:
        score = 0.99
        feedback = (
            "Perfect: gestational_diabetes correctly identified with visit_phc_this_week urgency."
        )
    elif predicted == EXPECTED_CONDITION and urgency == "go_to_hospital_today":
        score = 0.8
        feedback = (
            "Condition correct. Hospital escalation is cautious but higher than needed without danger flags."
        )
    elif predicted == EXPECTED_CONDITION and urgency == "monitor_at_home":
        score = 0.25
        feedback = (
            "Condition correct, but monitor_at_home under-escalates a patient who needs clinical follow-up."
        )
    elif predicted == "anemia" and urgency == EXPECTED_URGENCY:
        score = 0.4
        feedback = (
            "Anemia is adjacent because low energy and breathlessness overlap, but family_diabetes and high meals make gestational_diabetes primary."
        )
    elif urgency == EXPECTED_URGENCY:
        score = 0.3
        feedback = (
            f"Urgency was appropriate, but condition '{predicted}' missed the metabolic pattern."
        )
    else:
        score = 0.01
        feedback = (
            f"Incorrect diagnosis ('{predicted}') and urgency ('{urgency}'). Expected gestational_diabetes / visit_phc_this_week."
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
