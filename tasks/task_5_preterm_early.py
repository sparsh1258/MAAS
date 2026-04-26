from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 405,
        "weeks_pregnant": 32,
        "trimester": 3,
        "region": "rural Rajasthan",
        "regional_access_tier": "rural",
        "risk_flags": ["DANGER_BLEEDING", "ABDOMINAL_PAIN_SIGNAL", "PRETERM_CONTRACTION_PROXY"],
        "composite_risk_score": 0.86,
        "bp_trend": "stable",
        "avg_kick_count": 6.0,
        "avg_meals": 2.0,
        "avg_sleep": 5.0,
        "latest_weight_kg": 60.0,
        "latest_energy": 5,
        "latest_breathlessness": 4,
        "symptom_cluster": ["bleeding", "abdominal_pain", "dizziness"],
        "bp_systolic_latest": 128,
        "bp_diastolic_latest": 84,
        "history_flags": ["prev_complication"],
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
        "id": "task_5_preterm_early",
        "description": "32-week bleeding + abdominal pain: early preterm risk triage with rural access constraints.",
        "difficulty": "hard",
        "observation": observation,
        "expected_condition": "preterm_risk",
        "expected_urgency": "go_to_hospital_today",
        "scoring_rubric": {
            "full_credit": "Preterm risk and hospital today.",
            "partial_credit": "Preeclampsia hospital today only if rationale anchors on BP/vision (not present).",
            "fail": "Low risk, anemia, or PHC-this-week urgency.",
        },
    }

