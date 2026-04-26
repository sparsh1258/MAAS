from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 408,
        "weeks_pregnant": 34,
        "trimester": 3,
        "region": "rural Rajasthan",
        "regional_access_tier": "rural",
        "risk_flags": ["HIGH_BP", "HIGH_PREECLAMPSIA_SIGNAL"],
        "composite_risk_score": 0.52,
        "bp_trend": "rising",
        "avg_kick_count": 7.0,
        "avg_meals": 2.2,
        "avg_sleep": 5.2,
        "latest_weight_kg": 74.0,
        "latest_energy": 4,
        "latest_breathlessness": 5,
        "symptom_cluster": ["headache", "swelling"],
        "bp_systolic_latest": 152,
        "bp_diastolic_latest": 96,
        "history_flags": ["family_hypertension"],
        "days_of_data": 3,
        "masked_signals": [],
        "episode_day_index": 1,
        "total_episode_days": 1,
        "belief_state": {},
        "available_signals": [],
        "withheld_signals": [],
        "signal_mask": {},
    }
    return {
        "id": "task_8_rural_access",
        "description": "Moderate preeclampsia signals in a rural access tier; triage should bias toward safer escalation planning.",
        "difficulty": "expert",
        "observation": observation,
        "expected_condition": "preeclampsia",
        "expected_urgency": "go_to_hospital_today",
        "scoring_rubric": {
            "full_credit": "Preeclampsia + hospital today (rural access and rising BP).",
            "partial_credit": "Preeclampsia + PHC-this-week with strong warning signs and contingency plan.",
            "fail": "Low risk or non-preeclampsia.",
        },
    }

