from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 409,
        "weeks_pregnant": 35,
        "trimester": 3,
        "region": "coastal Odisha",
        "regional_access_tier": "semi_urban",
        "risk_flags": [
            "DANGER_BP_CRITICAL",
            "DANGER_VISION_HEADACHE",
            "SYMPTOM_CLUSTER_HIGH",
            "VOMITING_SIGNAL",
            "MATERNAL_EXHAUSTION",
        ],
        "composite_risk_score": 0.92,
        "bp_trend": "rising",
        "avg_kick_count": 5.0,
        "avg_meals": 1.8,
        "avg_sleep": 3.8,
        "latest_weight_kg": 79.0,
        "latest_energy": 2,
        "latest_breathlessness": 7,
        "symptom_cluster": ["headache", "blurred_vision", "vomiting", "abdominal_pain"],
        "bp_systolic_latest": 172,
        "bp_diastolic_latest": 114,
        "history_flags": ["prev_preeclampsia"],
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
        "id": "task_9_hellp_proxy",
        "description": "HELLP-like proxy: severe BP + vision/headache + vomiting/exhaustion; must prioritize immediate hospital care.",
        "difficulty": "expert",
        "observation": observation,
        "expected_condition": "preeclampsia",
        "expected_urgency": "go_to_hospital_today",
        "scoring_rubric": {
            "full_credit": "Preeclampsia + hospital today.",
            "partial_credit": "Preeclampsia + PHC-this-week (unsafe but recognizes diagnosis).",
            "fail": "Anything else.",
        },
    }

