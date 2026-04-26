from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 413,
        "weeks_pregnant": 27,
        "trimester": 2,
        "region": "urban Karnataka",
        "regional_access_tier": "urban",
        # Contradictory/noisy: one danger-ish symptom but normal-ish vitals.
        "risk_flags": ["DIZZINESS_SIGNAL", "ABDOMINAL_PAIN_SIGNAL"],
        "composite_risk_score": 0.34,
        "bp_trend": "stable",
        "avg_kick_count": 8.0,
        "avg_meals": 3.6,
        "avg_sleep": 7.8,
        "latest_weight_kg": 70.0,
        "latest_energy": 6,
        "latest_breathlessness": 3,
        "symptom_cluster": ["dizziness", "abdominal_pain"],
        "bp_systolic_latest": 118,
        "bp_diastolic_latest": 76,
        "history_flags": ["family_diabetes", "family_hypertension"],
        "days_of_data": 1,
        "masked_signals": [],
        "episode_day_index": 1,
        "total_episode_days": 1,
        "belief_state": {},
        "available_signals": [],
        "withheld_signals": [],
        "signal_mask": {},
    }
    return {
        "id": "task_13_adversarial_noise",
        "description": "Adversarial noise: contradictory signals and low data confidence; do not overfit one flag.",
        "difficulty": "expert",
        "observation": observation,
        "expected_condition": "preterm_risk",
        "expected_urgency": "visit_phc_this_week",
        "scoring_rubric": {
            "full_credit": "Preterm risk with PHC-this-week and request/monitor plan due to abdominal pain.",
            "partial_credit": "Anemia PHC-this-week if rationale ties dizziness and data uncertainty.",
            "fail": "Low risk with home monitoring; hospital today without any danger flags.",
        },
    }

