"""
Task 8 - MEDIUM: Escalating Preeclampsia Without Critical BP Crisis
===================================================================
Objective:
  Detect preeclampsia when headache/swelling and rising BP are present, even
  before a DANGER_BP_CRITICAL reading appears.
"""

from environment import Observation

TASK_ID = "task_8_preeclampsia_watch"
TASK_NAME = "Escalating Preeclampsia Watch"
TASK_DESCRIPTION = (
    "Detect preeclampsia from rising BP and classic symptoms before a hypertensive crisis appears."
)

TASK_OBSERVATION = Observation(
    user_id=9008,
    weeks_pregnant=32,
    trimester=3,
    region="Bihar",
    risk_flags=["HIGH_BP", "HIGH_PREECLAMPSIA_SIGNAL", "BP_RISING_TREND"],
    bp_trend="rising",
    avg_kick_count=8.6,
    avg_meals=2.7,
    avg_sleep=5.6,
    latest_weight_kg=74.0,
    latest_energy=4,
    latest_breathlessness=6,
    history_flags=["family_hypertension", "prev_preeclampsia"],
    days_of_data=3,
)

EXPECTED_CONDITION = "preeclampsia"
EXPECTED_URGENCY = "go_to_hospital_today"


def grade(action: dict) -> dict:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    if predicted == EXPECTED_CONDITION and urgency == EXPECTED_URGENCY:
        score = 0.99
        feedback = "Perfect: preeclampsia correctly identified and urgently escalated."
    elif predicted == EXPECTED_CONDITION and urgency == "visit_phc_this_week":
        score = 0.35
        feedback = "Condition is right, but the symptom cluster and rising BP require hospital escalation."
    elif predicted in {"gestational_diabetes", "anemia"} and urgency == EXPECTED_URGENCY:
        score = 0.2
        feedback = "Urgency is safe, but the dominant signal is preeclampsia, not a metabolic or anemia pattern."
    elif urgency == EXPECTED_URGENCY:
        score = 0.3
        feedback = f"Urgency was correct, but condition '{predicted}' was not the best fit."
    else:
        score = 0.01
        feedback = (
            f"Incorrect diagnosis ('{predicted}') and urgency ('{urgency}'). "
            "Expected preeclampsia / go_to_hospital_today."
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
