"""
Task 7 - HARD: Preterm Risk With Bleeding
=========================================
Objective:
  Detect preterm_risk when bleeding and abdominal pain dominate the episode and
  immediate hospital escalation is required.
"""

from environment import Observation

TASK_ID = "task_7_preterm_risk"
TASK_NAME = "Preterm Risk With Bleeding"
TASK_DESCRIPTION = (
    "Detect preterm_risk from bleeding and abdominal pain in late pregnancy and escalate to hospital."
)

TASK_OBSERVATION = Observation(
    user_id=9007,
    weeks_pregnant=33,
    trimester=3,
    region="Chhattisgarh",
    risk_flags=["DANGER_BLEEDING", "ABDOMINAL_PAIN_SIGNAL", "DIZZINESS_SIGNAL"],
    bp_trend="stable",
    avg_kick_count=7.1,
    avg_meals=2.4,
    avg_sleep=4.9,
    latest_weight_kg=60.0,
    latest_energy=3,
    latest_breathlessness=5,
    history_flags=["prev_complication"],
    days_of_data=3,
)

EXPECTED_CONDITION = "preterm_risk"
EXPECTED_URGENCY = "go_to_hospital_today"


def grade(action: dict) -> dict:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    if predicted == EXPECTED_CONDITION and urgency == EXPECTED_URGENCY:
        score = 0.99
        feedback = "Perfect: preterm_risk correctly identified and urgently escalated."
    elif predicted == "fetal_distress" and urgency == EXPECTED_URGENCY:
        score = 0.35
        feedback = "Urgency is correct, but bleeding and abdominal pain make preterm_risk primary here."
    elif urgency == "visit_phc_this_week":
        score = 0.12
        feedback = "Under-escalated: active bleeding requires go_to_hospital_today."
    elif urgency == "monitor_at_home":
        score = 0.01
        feedback = "Critical failure: monitor_at_home is unsafe with DANGER_BLEEDING."
    elif urgency == EXPECTED_URGENCY:
        score = 0.3
        feedback = f"Urgency was correct, but condition '{predicted}' was not."
    else:
        score = 0.01
        feedback = (
            f"Incorrect diagnosis ('{predicted}') and urgency ('{urgency}'). Expected preterm_risk / go_to_hospital_today."
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
