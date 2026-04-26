from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 412,
        "weeks_pregnant": 33,
        "trimester": 3,
        "region": "coastal Odisha",
        "regional_access_tier": "semi_urban",
        "risk_flags": ["BP_RISING_TREND", "LOW_NUTRITION"],
        "composite_risk_score": 0.36,
        "bp_trend": "rising",
        "avg_kick_count": 7.0,
        "avg_meals": 1.9,
        "avg_sleep": 5.5,
        "latest_weight_kg": 63.0,
        "latest_energy": 4,
        "latest_breathlessness": 5,
        "symptom_cluster": ["headache"],
        "bp_systolic_latest": 138,
        "bp_diastolic_latest": 88,
        "history_flags": ["prev_complication"],
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
        "id": "task_12_multiparous_risk",
        "description": "Multiparous risk: previous complication + late pregnancy + rising BP trend; cautious triage.",
        "difficulty": "hard",
        "observation": observation,
        "expected_condition": "preeclampsia",
        "expected_urgency": "visit_phc_this_week",
        "scoring_rubric": {
            "full_credit": "Preeclampsia spectrum with PHC-this-week and clear escalation criteria.",
            "partial_credit": "Hospital today if rationale emphasizes previous complication + rising trend.",
            "fail": "Low risk or gestational diabetes.",
        },
    }

