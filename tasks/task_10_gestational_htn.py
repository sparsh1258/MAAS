from __future__ import annotations


def get_task() -> dict:
    observation = {
        "user_id": 410,
        "weeks_pregnant": 29,
        "trimester": 3,
        "region": "urban Karnataka",
        "regional_access_tier": "urban",
        "risk_flags": ["HIGH_BP"],
        "composite_risk_score": 0.28,
        "bp_trend": "stable",
        "avg_kick_count": 8.0,
        "avg_meals": 3.0,
        "avg_sleep": 7.2,
        "latest_weight_kg": 68.0,
        "latest_energy": 6,
        "latest_breathlessness": 3,
        "symptom_cluster": [],
        "bp_systolic_latest": 144,
        "bp_diastolic_latest": 92,
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
        "id": "task_10_gestational_htn",
        "description": "Gestational hypertension: high BP without other symptoms; monitor and follow-up appropriately.",
        "difficulty": "medium",
        "observation": observation,
        "expected_condition": "preeclampsia",
        "expected_urgency": "visit_phc_this_week",
        "scoring_rubric": {
            "full_credit": "Preeclampsia spectrum + PHC-this-week with warning signs to escalate if worsens.",
            "partial_credit": "Hospital today if rationale cites access issues or rising trend (not shown).",
            "fail": "Low risk or ignoring high BP.",
        },
    }

