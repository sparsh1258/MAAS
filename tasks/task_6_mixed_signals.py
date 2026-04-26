from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 406,
        "weeks_pregnant": 30,
        "trimester": 3,
        "region": "coastal Odisha",
        "regional_access_tier": "semi_urban",
        "risk_flags": ["HIGH_BP", "BP_RISING_TREND"],
        "composite_risk_score": 0.44,
        "bp_trend": "rising",
        "avg_kick_count": 7.5,
        "avg_meals": 3.4,
        "avg_sleep": 6.2,
        "latest_weight_kg": 72.0,
        "latest_energy": 4,
        "latest_breathlessness": 5,
        "symptom_cluster": ["dizziness"],
        "bp_systolic_latest": 148,
        "bp_diastolic_latest": 94,
        "history_flags": ["family_diabetes"],
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
        "id": "task_6_mixed_signals",
        "description": "Ambiguous: diabetes history + rising BP. Avoid anchoring; prioritize safety if BP trend persists.",
        "difficulty": "expert",
        "observation": observation,
        "expected_condition": "preeclampsia",
        "expected_urgency": "visit_phc_this_week",
        "scoring_rubric": {
            "full_credit": "Preeclampsia with PHC-this-week (no critical danger flags yet) and clear safety rationale.",
            "partial_credit": "Preeclampsia hospital today if rationale cites uncertainty + worsening signs.",
            "fail": "Gestational diabetes primary label without addressing rising BP; low risk.",
        },
    }

