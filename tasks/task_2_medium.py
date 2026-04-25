"""
Task 2 — MEDIUM: Preeclampsia Detection
========================================
Objective:
  Identify preeclampsia in a 3rd-trimester patient with elevated BP,
  rising BP trend, headache + swelling symptoms, and a history of
  hypertension. Urgency must be visit_phc_this_week or higher.

Difficulty: Medium
  - Multiple signals pointing to preeclampsia but no DANGER-level flags
  - Competing signals could suggest gestational_diabetes (history_diabetes)
  - Agent must weigh BP + history correctly

Expected output:
  condition = preeclampsia
  urgency   = visit_phc_this_week  (or go_to_hospital_today for extra caution)
  score     = 1.0 for exact match, 0.8 for correct condition + higher urgency

Partial credit rubric:
  correct condition + correct urgency:        1.0
  correct condition + go_to_hospital_today:   0.8  (cautious, acceptable)
  correct condition + monitor_at_home:        0.3  (dangerous under-escalation)
  wrong condition + correct urgency level:    0.3
  preterm_risk (adjacent) + right urgency:    0.4
  both wrong:                                 0.0
"""

from environment import Observation

TASK_ID = "task_2_medium"
TASK_NAME = "Preeclampsia Detection"
TASK_DESCRIPTION = (
    "Detect preeclampsia in a 3rd-trimester patient showing elevated BP, "
    "rising trend, headache + swelling, and hypertension history. "
    "Set appropriate escalation urgency."
)

TASK_OBSERVATION = Observation(
    user_id=9002,
    weeks_pregnant=34,
    trimester=3,
    region="Bihar",
    risk_flags=["HIGH_BP", "BP_RISING_TREND", "HIGH_PREECLAMPSIA_SIGNAL"],
    bp_trend="rising",
    avg_kick_count=8.5,
    avg_meals=3.2,
    avg_sleep=6.0,
    latest_weight_kg=74.0,
    latest_energy=4,
    latest_breathlessness=5,
    history_flags=["family_hypertension"],
    days_of_data=3,
)

EXPECTED_CONDITION = "preeclampsia"
EXPECTED_URGENCY = "visit_phc_this_week"
ACCEPTABLE_URGENCIES = {"visit_phc_this_week", "go_to_hospital_today"}
ADJACENT_CONDITIONS = {"preterm_risk"}


def grade(action: dict) -> dict:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    correct_condition = predicted == EXPECTED_CONDITION
    acceptable_urgency = urgency in ACCEPTABLE_URGENCIES
    under_escalated = urgency == "monitor_at_home"

    if correct_condition and urgency == EXPECTED_URGENCY:
        score = 0.99
        feedback = (
            "Perfect: preeclampsia correctly identified with visit_phc_this_week urgency. "
            "Rising BP, headache+swelling, and hypertension history all point to this."
        )
    elif correct_condition and urgency == "go_to_hospital_today":
        score = 0.8
        feedback = (
            "Condition correct (preeclampsia). Urgency 'go_to_hospital_today' is an "
            "over-escalation — BP is elevated but not yet critical (no DANGER_BP_CRITICAL). "
            "visit_phc_this_week was sufficient here, but this is a safe call."
        )
    elif correct_condition and under_escalated:
        score = 0.3
        feedback = (
            "Condition correct (preeclampsia) but 'monitor_at_home' is dangerous under-escalation. "
            "A patient with HIGH_BP + rising trend + hypertension history needs PHC evaluation."
        )
    elif predicted in ADJACENT_CONDITIONS and acceptable_urgency:
        score = 0.4
        feedback = (
            f"Condition '{predicted}' is adjacent (shares some risk signals) but preeclampsia "
            "was the primary diagnosis. BP and swelling+headache signals outweigh preterm signals. "
            f"Urgency '{urgency}' was appropriate."
        )
    elif not correct_condition and acceptable_urgency:
        score = 0.3
        feedback = (
            f"Wrong condition ('{predicted}'), but urgency '{urgency}' was appropriate. "
            "Review: HIGH_BP + BP_RISING_TREND + HIGH_PREECLAMPSIA_SIGNAL + family_hypertension "
            "strongly indicate preeclampsia."
        )
    else:
        score = 0.01
        feedback = (
            f"Both condition ('{predicted}') and urgency ('{urgency}') are incorrect. "
            "Key signals: HIGH_BP, BP_RISING_TREND, HIGH_PREECLAMPSIA_SIGNAL (headache+swelling), "
            "family_hypertension history, 3rd trimester. Expected: preeclampsia / visit_phc_this_week."
        )

    return {
        "score": round(score, 4),
        "passed": score >= 0.99,
        "feedback": feedback,
        "expected": {
            "condition": EXPECTED_CONDITION,
            "urgency": EXPECTED_URGENCY,
            "also_acceptable_urgency": "go_to_hospital_today",
        },
        "predicted": {"condition": predicted, "urgency": urgency},
    }


def get_task_prompt() -> str:
    obs = TASK_OBSERVATION
    return f"""
You are Niva, an AI maternal health assistant working in rural India.

A pregnant woman's recent health data requires your assessment.
Multiple warning signals are present — analyze them carefully.

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

=== CLINICAL CONTEXT ===
HIGH_PREECLAMPSIA_SIGNAL means the patient reported BOTH headache AND swelling.
BP_RISING_TREND means systolic rose >10 mmHg over the last 3 readings.
HIGH_BP means latest reading was >=140/90 but below critical threshold (160/110).

=== YOUR TASK ===
Call the diagnose action with:
  - condition: one of [preeclampsia, gestational_diabetes, anemia, preterm_risk, fetal_distress, low_risk]
  - urgency: one of [monitor_at_home, visit_phc_this_week, go_to_hospital_today]
  - rationale: a short clinical explanation

Respond in JSON:
{{"condition": "<condition>", "urgency": "<urgency>", "rationale": "<short explanation>"}}
""".strip()
