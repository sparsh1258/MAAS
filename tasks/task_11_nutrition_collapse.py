from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 411,
        "weeks_pregnant": 24,
        "trimester": 2,
        "region": "tribal Jharkhand",
        "regional_access_tier": "rural",
        "risk_flags": ["LOW_NUTRITION", "SEVERE_ANEMIA_PROXY"],
        "composite_risk_score": 0.68,
        "bp_trend": "stable",
        "avg_kick_count": 6.5,
        "avg_meals": 0.9,
        "avg_sleep": 5.0,
        "latest_weight_kg": 48.0,
        "latest_energy": 2,
        "latest_breathlessness": 8,
        "symptom_cluster": ["dizziness"],
        "bp_systolic_latest": 104,
        "bp_diastolic_latest": 68,
        "history_flags": [],
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
        "id": "task_11_nutrition_collapse",
        "description": "Severe anemia from starvation-level meals; prioritize urgent correction and safety planning.",
        "difficulty": "hard",
        "observation": observation,
        "expected_condition": "anemia",
        "expected_urgency": "go_to_hospital_today",
        "scoring_rubric": {
            "full_credit": "Anemia + hospital today.",
            "partial_credit": "Anemia + PHC-this-week with explicit urgent red flags.",
            "fail": "Low risk or non-anemia.",
        },
    }

