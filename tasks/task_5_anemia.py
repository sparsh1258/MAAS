"""
Task 5 - MEDIUM: Maternal Anemia Identification
===============================================
Objective:
  Identify anemia from poor nutrition, dizziness, low energy, and
  breathlessness without immediate bleeding or blood-pressure danger flags.
"""

from environment import Observation

TASK_ID = "task_5_anemia"
TASK_NAME = "Maternal Anemia Identification"
TASK_DESCRIPTION = (
    "Detect anemia from low nutrition and fatigue signals and assign outpatient follow-up urgency."
)

TASK_OBSERVATION = Observation(
    user_id=9005,
    weeks_pregnant=27,
    trimester=3,
    region="Jharkhand",
    risk_flags=["LOW_NUTRITION", "DIZZINESS_SIGNAL"],
    bp_trend="stable",
    avg_kick_count=8.8,
    avg_meals=1.7,
    avg_sleep=5.0,
    latest_weight_kg=51.0,
    latest_energy=3,
    latest_breathlessness=7,
    history_flags=[],
    days_of_data=3,
)

EXPECTED_CONDITION = "anemia"
EXPECTED_URGENCY = "visit_phc_this_week"


def grade(action: dict) -> dict:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    if predicted == EXPECTED_CONDITION and urgency == EXPECTED_URGENCY:
        score = 0.99
        feedback = "Perfect: anemia correctly identified with outpatient follow-up urgency."
    elif predicted == EXPECTED_CONDITION and urgency == "go_to_hospital_today":
        score = 0.75
        feedback = "Condition correct. Hospital escalation is safe but stronger than needed here."
    elif predicted == EXPECTED_CONDITION and urgency == "monitor_at_home":
        score = 0.25
        feedback = "Condition correct, but monitor_at_home misses the need for timely nutrition and anemia follow-up."
    elif predicted == "gestational_diabetes" and urgency == EXPECTED_URGENCY:
        score = 0.3
        feedback = "Gestational diabetes is not primary here; low_nutrition and dizziness make anemia the better fit."
    elif urgency == EXPECTED_URGENCY:
        score = 0.3
        feedback = f"Urgency was acceptable, but condition '{predicted}' was incorrect."
    else:
        score = 0.01
        feedback = (
            f"Incorrect diagnosis ('{predicted}') and urgency ('{urgency}'). Expected anemia / visit_phc_this_week."
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
