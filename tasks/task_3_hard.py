"""
Task 3 — HARD: Emergency Triage — Competing Conditions + Danger Flags
======================================================================
Objective:
  Handle a 3rd-trimester patient with DANGER-level flags (critical BP + low
  kicks) alongside competing signals for fetal_distress AND preeclampsia.
  Agent must:
    1. Correctly identify the MOST DANGEROUS primary condition
    2. Correctly set urgency to go_to_hospital_today (the only safe choice)
    3. Resist the temptation to diagnose the secondary condition as primary

Difficulty: Hard
  - DANGER_BP_CRITICAL overrides should force urgency = go_to_hospital_today
  - DANGER_LOW_KICKS simultaneously present (fetal_distress signal)
  - Agent must decide: preeclampsia or fetal_distress?
  - Full history (diabetes + hypertension + prev_preeclampsia) adds noise
  - Under-escalating urgency = catastrophic error
  - Diagnosing anemia or gestational_diabetes = shows misunderstanding

Scoring rubric:
  preeclampsia + go_to_hospital_today:          1.0   (ideal: DANGER_BP_CRITICAL is primary)
  fetal_distress + go_to_hospital_today:        0.9   (also valid: DANGER_LOW_KICKS also present)
  preterm_risk + go_to_hospital_today:          0.5   (wrong condition, right urgency — partial)
  any condition + visit_phc_this_week:          0.2   (dangerous under-escalation)
  any condition + monitor_at_home:              0.0   (critical safety failure)
  gestational_diabetes or anemia (any urgency): 0.1   (completely wrong read)
  low_risk (any urgency):                       0.0   (disqualifying error)
"""

from environment import Observation

TASK_ID = "task_3_hard"
TASK_NAME = "Emergency Triage — Competing Danger Signals"
TASK_DESCRIPTION = (
    "Triage a 3rd-trimester patient with DANGER_BP_CRITICAL and DANGER_LOW_KICKS "
    "simultaneously. Competing signals exist for both preeclampsia and fetal_distress. "
    "Agent must correctly escalate to go_to_hospital_today and identify the primary condition."
)

TASK_OBSERVATION = Observation(
    user_id=9003,
    weeks_pregnant=38,
    trimester=3,
    region="Uttar Pradesh",
    risk_flags=[
        "DANGER_BP_CRITICAL",       # BP >= 160/110
        "DANGER_LOW_KICKS",         # kick_count < 3
        "HIGH_PREECLAMPSIA_SIGNAL", # headache + swelling
        "BP_RISING_TREND",
    ],
    bp_trend="rising",
    avg_kick_count=2.3,             # dangerously low
    avg_meals=2.1,
    avg_sleep=4.5,                  # poor sleep
    latest_weight_kg=81.0,
    latest_energy=2,                # very low
    latest_breathlessness=8,        # very high
    history_flags=[
        "family_diabetes",
        "family_hypertension",
        "prev_preeclampsia",
        "prev_complication",
    ],
    days_of_data=3,
)

PRIMARY_EXPECTED = "preeclampsia"          # DANGER_BP_CRITICAL is the top-priority flag
ALSO_VALID = "fetal_distress"              # DANGER_LOW_KICKS is also a real danger
REQUIRED_URGENCY = "go_to_hospital_today"

# Conditions that show complete misreading
DISQUALIFYING_CONDITIONS = {"low_risk", "gestational_diabetes", "anemia"}


def grade(action: dict) -> dict:
    predicted = action.get("condition") or action.get("target")
    urgency = action.get("urgency")

    correct_urgency = urgency == REQUIRED_URGENCY
    is_primary = predicted == PRIMARY_EXPECTED
    is_also_valid = predicted == ALSO_VALID
    is_adjacent = predicted == "preterm_risk"
    is_disqualifying = predicted in DISQUALIFYING_CONDITIONS

    # ── Urgency is non-negotiable for safety ───────────────
    if not correct_urgency:
        if urgency == "visit_phc_this_week":
            score = 0.2
            feedback = (
                f"DANGEROUS UNDER-ESCALATION: '{urgency}' for a patient with DANGER_BP_CRITICAL "
                "and DANGER_LOW_KICKS is clinically unsafe. Both flags mandate immediate hospital. "
                f"Condition '{predicted}' {'was correct' if is_primary or is_also_valid else 'was also wrong'}."
            )
        else:  # monitor_at_home
            score = 0.01
            feedback = (
                "CRITICAL SAFETY FAILURE: 'monitor_at_home' with active DANGER flags "
                "(DANGER_BP_CRITICAL + DANGER_LOW_KICKS) could cost the patient's life. "
                "Any DANGER_ flag mandates go_to_hospital_today."
            )
        return {
            "score": round(score, 4),
            "passed": False,
            "feedback": feedback,
            "expected": {"condition": PRIMARY_EXPECTED, "urgency": REQUIRED_URGENCY},
            "predicted": {"condition": predicted, "urgency": urgency},
        }

    # ── Urgency is correct (go_to_hospital_today) ──────────
    if is_primary:
        score = 0.99
        feedback = (
            "Perfect: preeclampsia correctly identified as primary (DANGER_BP_CRITICAL is the "
            "highest-priority flag) with go_to_hospital_today. "
            "Note: fetal_distress was also present but BP crisis takes clinical precedence."
        )
    elif is_also_valid:
        score = 0.9
        feedback = (
            "Near-perfect: fetal_distress is a valid primary diagnosis (DANGER_LOW_KICKS present) "
            "with correct urgency go_to_hospital_today. "
            "Preeclampsia is the preferred primary as DANGER_BP_CRITICAL is the overriding flag, "
            "but this is clinically defensible."
        )
    elif is_adjacent:
        score = 0.5
        feedback = (
            "Condition 'preterm_risk' is adjacent but misses the primary diagnosis. "
            "DANGER_BP_CRITICAL and DANGER_LOW_KICKS are the urgent signals — preterm_risk "
            "is a secondary concern here. Urgency was correct."
        )
    elif is_disqualifying:
        score = 0.1
        feedback = (
            f"Condition '{predicted}' shows a fundamental misread of the observation. "
            "DANGER_BP_CRITICAL, DANGER_LOW_KICKS, and HIGH_PREECLAMPSIA_SIGNAL clearly dominate. "
            "Urgency was correct, which prevents a score of 0."
        )
    else:
        # Other wrong condition with correct urgency
        score = 0.3
        feedback = (
            f"Condition '{predicted}' is incorrect but urgency was correct. "
            "Primary signals: DANGER_BP_CRITICAL → preeclampsia, DANGER_LOW_KICKS → fetal_distress. "
            "With full history (prev_preeclampsia, family_hypertension), preeclampsia is primary."
        )

    return {
        "score": round(score, 4),
        "passed": score >= 0.99,
        "feedback": feedback,
        "expected": {
            "condition": PRIMARY_EXPECTED,
            "also_valid_condition": ALSO_VALID,
            "urgency": REQUIRED_URGENCY,
            "note": "Urgency is non-negotiable with active DANGER flags.",
        },
        "predicted": {"condition": predicted, "urgency": urgency},
    }


def get_task_prompt() -> str:
    obs = TASK_OBSERVATION
    return f"""
You are Niva, an AI maternal health assistant working in rural India.

URGENT: This patient has active DANGER-level flags. Analyze carefully.

=== PATIENT OBSERVATION ===
Weeks Pregnant   : {obs.weeks_pregnant} (Trimester {obs.trimester})
Region           : {obs.region}
Risk Flags       : {obs.risk_flags}
BP Trend         : {obs.bp_trend}
Avg Kick Count   : {obs.avg_kick_count} (CRITICALLY LOW — normal is >=10/hour)
Avg Meals/Day    : {obs.avg_meals}
Avg Sleep Hours  : {obs.avg_sleep}
Latest Weight    : {obs.latest_weight_kg} kg
Energy Level     : {obs.latest_energy}/10
Breathlessness   : {obs.latest_breathlessness}/10
Medical History  : {obs.history_flags}
Days of Data     : {obs.days_of_data}

=== DANGER FLAG LEGEND ===
DANGER_BP_CRITICAL        → BP >= 160/110 (hypertensive crisis)
DANGER_LOW_KICKS          → Kick count < 3 in the last session (fetal emergency)
HIGH_PREECLAMPSIA_SIGNAL  → Patient reported BOTH headache AND swelling
BP_RISING_TREND           → Systolic rose >10 mmHg over 3 consecutive readings

=== YOUR TASK ===
Multiple danger conditions are present simultaneously.
Determine the PRIMARY condition and appropriate urgency.

Call the diagnose action with:
  - condition: one of [preeclampsia, gestational_diabetes, anemia, preterm_risk, fetal_distress, low_risk]
  - urgency: one of [monitor_at_home, visit_phc_this_week, go_to_hospital_today]
  - rationale: a short clinical explanation

Respond in JSON:
{{"condition": "<condition>", "urgency": "<urgency>", "rationale": "<short explanation>"}}
""".strip()
