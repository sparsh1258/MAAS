from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 415,
        "weeks_pregnant": 36,
        "trimester": 3,
        "region": "rural Rajasthan",
        "regional_access_tier": "rural",
        "risk_flags": [
            "DANGER_BP_CRITICAL",
            "DANGER_LOW_KICKS",
            "DANGER_BLEEDING",
            "DANGER_VISION_HEADACHE",
            "SYMPTOM_CLUSTER_HIGH",
            "MATERNAL_EXHAUSTION",
        ],
        "composite_risk_score": 1.0,
        "bp_trend": "rising",
        "avg_kick_count": 1.0,
        "avg_meals": 1.4,
        "avg_sleep": 3.5,
        "latest_weight_kg": 82.0,
        "latest_energy": 2,
        "latest_breathlessness": 8,
        "symptom_cluster": ["headache", "blurred_vision", "bleeding", "abdominal_pain", "dizziness"],
        "bp_systolic_latest": 178,
        "bp_diastolic_latest": 120,
        "history_flags": ["prev_preeclampsia", "prev_complication"],
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
        "id": "task_15_expert_triage",
        "description": "Expert triage: multiple danger flags firing simultaneously; agent must prioritize immediate hospital escalation.",
        "difficulty": "expert",
        "observation": observation,
        "expected_condition": "preeclampsia",
        "expected_urgency": "go_to_hospital_today",
        "scoring_rubric": {
            "full_credit": "Hospital today and choose a danger condition (preeclampsia preferred given BP/vision/headache).",
            "partial_credit": "Fetal distress hospital today (acknowledges emergency but misses primary maternal danger).",
            "fail": "Anything below hospital today.",
        },
    }

