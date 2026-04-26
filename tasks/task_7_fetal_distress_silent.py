from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 407,
        "weeks_pregnant": 36,
        "trimester": 3,
        "region": "urban Karnataka",
        "regional_access_tier": "urban",
        "risk_flags": ["DANGER_LOW_KICKS"],
        "composite_risk_score": 0.55,
        "bp_trend": "stable",
        "avg_kick_count": 0.0,
        "avg_meals": 3.0,
        "avg_sleep": 7.0,
        "latest_weight_kg": 66.0,
        "latest_energy": 6,
        "latest_breathlessness": 3,
        "symptom_cluster": [],
        "bp_systolic_latest": 122,
        "bp_diastolic_latest": 78,
        "history_flags": [],
        "days_of_data": 2,
        "masked_signals": [],
        "episode_day_index": 1,
        "total_episode_days": 1,
        "belief_state": {},
        "available_signals": [],
        "withheld_signals": [],
        "signal_mask": {},
    }
    return {
        "id": "task_7_fetal_distress_silent",
        "description": "Zero kicks with otherwise quiet symptoms; must treat as fetal distress emergency.",
        "difficulty": "hard",
        "observation": observation,
        "expected_condition": "fetal_distress",
        "expected_urgency": "go_to_hospital_today",
        "scoring_rubric": {
            "full_credit": "Fetal distress + hospital today.",
            "partial_credit": "Fetal distress + PHC-this-week (still unsafe but recognizes condition).",
            "fail": "Low risk or any home-monitor urgency.",
        },
    }

