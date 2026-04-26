from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 404,
        "weeks_pregnant": 28,
        "trimester": 3,
        "region": "tribal Jharkhand",
        "regional_access_tier": "rural",
        "risk_flags": [
            "LOW_NUTRITION",
            "DIZZINESS_SIGNAL",
            "SEVERE_ANEMIA_PROXY",
            "MATERNAL_EXHAUSTION",
        ],
        "composite_risk_score": 0.72,
        "bp_trend": "stable",
        "avg_kick_count": 7.0,
        "avg_meals": 1.2,
        "avg_sleep": 3.6,
        "latest_weight_kg": 52.0,
        "latest_energy": 2,
        "latest_breathlessness": 9,
        "symptom_cluster": ["dizziness", "vomiting"],
        "bp_systolic_latest": 110,
        "bp_diastolic_latest": 72,
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
        "id": "task_4_anemia_severe",
        "description": "Hb-proxy crash: severe fatigue + breathlessness + dizziness + poor nutrition; avoid under-triage.",
        "difficulty": "hard",
        "observation": observation,
        "expected_condition": "anemia",
        "expected_urgency": "go_to_hospital_today",
        "scoring_rubric": {
            "full_credit": "Diagnose anemia and escalate to hospital today given severe proxy signals.",
            "partial_credit": "Anemia with PHC-this-week if rationale mentions close monitoring (still suboptimal).",
            "fail": "Low risk or non-anemia primary label; any home-monitor urgency.",
        },
    }

