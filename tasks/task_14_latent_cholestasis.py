from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 414,
        "weeks_pregnant": 38,
        "trimester": 3,
        "region": "coastal Odisha",
        "regional_access_tier": "semi_urban",
        # No explicit danger flags; late pregnancy discomfort; latent risk awareness.
        "risk_flags": [],
        "composite_risk_score": 0.10,
        "bp_trend": "stable",
        "avg_kick_count": 8.5,
        "avg_meals": 2.8,
        "avg_sleep": 6.2,
        "latest_weight_kg": 71.0,
        "latest_energy": 5,
        "latest_breathlessness": 4,
        "symptom_cluster": ["vomiting"],
        "bp_systolic_latest": 120,
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
        "id": "task_14_latent_cholestasis",
        "description": "Late pregnancy discomfort with possible latent cholestasis-like signal; avoid panic but schedule follow-up.",
        "difficulty": "medium",
        "observation": observation,
        "expected_condition": "low_risk",
        "expected_urgency": "visit_phc_this_week",
        "scoring_rubric": {
            "full_credit": "Low risk with PHC-this-week follow-up plan (late pregnancy + symptoms).",
            "partial_credit": "Gestational diabetes PHC-this-week if rationale ties late pregnancy and symptoms (still imperfect).",
            "fail": "Hospital today without danger flags; monitor at home without follow-up.",
        },
    }

