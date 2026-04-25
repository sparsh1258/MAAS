"""
Explainable reward model for the LLM triage agent.

This file contains the deterministic feature engineering and safety rules that
score an LLM's diagnosis and urgency decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


SAFE_CONDITIONS = [
    "preeclampsia",
    "gestational_diabetes",
    "anemia",
    "preterm_risk",
    "fetal_distress",
    "low_risk",
]

URGENCY_ORDER = ["monitor_at_home", "visit_phc_this_week", "go_to_hospital_today"]

CONDITION_SEVERITY = {
    "preeclampsia": 10,
    "fetal_distress": 10,
    "preterm_risk": 7,
    "gestational_diabetes": 6,
    "anemia": 5,
    "low_risk": 0,
}

FEATURE_CONDITION_RULES = {
    "preeclampsia": [
        "danger_bp",
        "high_bp",
        "bp_rising",
        "headache_swelling",
        "vision_headache",
        "history_htn",
        "history_preeclampsia",
    ],
    "gestational_diabetes": [
        "history_diabetes",
        "high_meals",
        "low_energy",
        "breathlessness",
        "trimester2_or_3",
        "weight_signal",
    ],
    "anemia": [
        "low_nutrition",
        "very_low_meals",
        "low_energy",
        "breathlessness",
        "dizziness",
        "poor_sleep",
    ],
    "preterm_risk": [
        "bleeding",
        "abdominal_pain",
        "history_complication",
        "weeks_early_third",
        "poor_sleep",
        "maternal_strain",
    ],
    "fetal_distress": [
        "danger_low_kicks",
        "low_kick_avg",
        "late_pregnancy",
        "maternal_strain",
    ],
    "low_risk": [
        "normal_bp",
        "good_kicks",
        "good_meals",
        "good_sleep",
        "no_history",
        "no_flags",
    ],
}


@dataclass(frozen=True)
class RewardBreakdown:
    predicted_condition: str
    reference_condition: str
    urgency: str
    reference_urgency: str
    reward: float
    reward_components: Dict[str, float]
    under_escalated: bool
    rationale: str
    latent_risks: Dict[str, float]
    supporting_features: List[str]


def featurize(obs) -> Dict[str, float]:
    flags = set(obs.risk_flags or [])
    history = set(obs.history_flags or [])
    avg_meals = obs.avg_meals or 0
    avg_sleep = obs.avg_sleep or 0
    kicks = obs.avg_kick_count
    energy = obs.latest_energy or 0
    breath = obs.latest_breathlessness or 0
    weeks = obs.weeks_pregnant or 0

    return {
        "danger_bp": float("DANGER_BP_CRITICAL" in flags),
        "high_bp": float("HIGH_BP" in flags),
        "bp_rising": float("BP_RISING_TREND" in flags or obs.bp_trend == "rising"),
        "headache_swelling": float("HIGH_PREECLAMPSIA_SIGNAL" in flags),
        "vision_headache": float("DANGER_VISION_HEADACHE" in flags),
        "danger_low_kicks": float("DANGER_LOW_KICKS" in flags),
        "low_kick_avg": float("LOW_KICK_AVG" in flags or (kicks is not None and kicks < 6)),
        "bleeding": float("DANGER_BLEEDING" in flags),
        "abdominal_pain": float("ABDOMINAL_PAIN_SIGNAL" in flags),
        "low_nutrition": float("LOW_NUTRITION" in flags),
        "very_low_meals": float(avg_meals and avg_meals < 2),
        "high_meals": float(avg_meals and avg_meals > 3.5),
        "good_meals": float(avg_meals >= 3),
        "poor_sleep": float(avg_sleep and avg_sleep < 5),
        "good_sleep": float(avg_sleep >= 7),
        "low_energy": float(energy and energy <= 4),
        "breathlessness": float(breath and breath >= 6),
        "maternal_strain": min(1.0, (max(0, 6 - energy) + max(0, breath - 4)) / 10.0),
        "history_diabetes": float("family_diabetes" in history),
        "history_htn": float("family_hypertension" in history),
        "history_preeclampsia": float("prev_preeclampsia" in history),
        "history_complication": float("prev_complication" in history),
        "no_history": float(not history),
        "trimester3": float(obs.trimester == 3),
        "trimester2_or_3": float(obs.trimester in (2, 3)),
        "weeks_early_third": float(obs.trimester == 3 and weeks < 37),
        "late_pregnancy": float(weeks >= 28),
        "weight_signal": float((obs.latest_weight_kg or 0) >= 75),
        "normal_bp": float(not ({"DANGER_BP_CRITICAL", "HIGH_BP"} & flags)),
        "good_kicks": float(kicks is not None and kicks >= 8),
        "no_flags": float(not flags),
        "dizziness": float("DIZZINESS_SIGNAL" in flags),
    }


def latent_risk_scores(features: Dict[str, float]) -> Dict[str, float]:
    scores = {
        "postpartum_hemorrhage": 0.65 * features["bleeding"] + 0.20 * features["low_energy"],
        "maternal_infection": 0.30 * features["abdominal_pain"] + 0.25 * features["dizziness"] + 0.20 * features["poor_sleep"],
        "dehydration": 0.45 * features["dizziness"] + 0.30 * features["maternal_strain"],
        "intrahepatic_cholestasis": 0.20 * features["late_pregnancy"] + 0.15 * features["maternal_strain"],
        "placental_abruption": 0.55 * features["bleeding"] + 0.35 * features["abdominal_pain"],
        "maternal_exhaustion": 0.35 * features["poor_sleep"] + 0.35 * features["low_energy"] + 0.25 * features["breathlessness"],
        "nutrition_deficit": 0.55 * features["low_nutrition"] + 0.25 * features["very_low_meals"],
    }
    return {k: round(min(0.99, v), 3) for k, v in scores.items() if v >= 0.25}


def infer_reference_condition(obs) -> str:
    features = featurize(obs)
    if features["danger_bp"] or features["headache_swelling"] or features["vision_headache"]:
        return "preeclampsia"
    if features["danger_low_kicks"] or features["low_kick_avg"]:
        return "fetal_distress"
    if features["bleeding"] or features["abdominal_pain"] or features["history_complication"]:
        return "preterm_risk"
    if features["history_diabetes"] and (features["high_meals"] or features["low_energy"] or features["breathlessness"]):
        return "gestational_diabetes"
    if features["low_nutrition"] or features["very_low_meals"] or features["dizziness"]:
        return "anemia"
    return "low_risk"


def supporting_features(condition: str, features: Dict[str, float]) -> List[str]:
    ranked = [name for name in FEATURE_CONDITION_RULES.get(condition, []) if features.get(name, 0.0) > 0]
    return ranked[:5] or ["baseline healthy pattern"]


def choose_urgency(condition: str, features: Dict[str, float]) -> str:
    if features["danger_bp"] or features["danger_low_kicks"] or features["bleeding"] or features["vision_headache"]:
        return "go_to_hospital_today"
    if condition in {"preeclampsia", "fetal_distress"}:
        return "go_to_hospital_today"
    if condition != "low_risk":
        return "visit_phc_this_week"
    return "monitor_at_home"


def calculate_reward(llm_diagnosis: str, llm_urgency: str, observation) -> RewardBreakdown:
    features = featurize(observation)
    reference_condition = infer_reference_condition(observation)
    reference_urgency = choose_urgency(reference_condition, features)
    urgency_idx = URGENCY_ORDER.index

    condition_score = 12.0 if llm_diagnosis == reference_condition else -6.0
    urgency_score = 0.0
    under_escalation_penalty = 0.0
    danger_override_penalty = 0.0
    data_recency_bonus = 0.0
    reward = condition_score

    under_escalated = False
    if llm_urgency == reference_urgency:
        urgency_score = 6.0
    elif urgency_idx(llm_urgency) > urgency_idx(reference_urgency):
        urgency_score = -2.0
    else:
        under_escalated = True
        under_escalation_penalty = -max(12.0, CONDITION_SEVERITY[reference_condition] * 2.5)

    if observation.days_of_data >= 3:
        data_recency_bonus = 2.0
    elif observation.days_of_data == 2:
        data_recency_bonus = 1.0

    if any(flag.startswith("DANGER") for flag in observation.risk_flags) and llm_urgency != "go_to_hospital_today":
        danger_override_penalty = -10.0
        under_escalated = True

    reward += urgency_score + under_escalation_penalty + danger_override_penalty + data_recency_bonus

    supporting = supporting_features(reference_condition, features)
    latent = latent_risk_scores(features)
    reward_components = {
        "condition_score": condition_score,
        "urgency_score": urgency_score,
        "under_escalation_penalty": under_escalation_penalty,
        "danger_override_penalty": danger_override_penalty,
        "data_recency_bonus": data_recency_bonus,
        "total_reward": reward,
    }
    rationale = (
        f"Reference condition={reference_condition}, reference urgency={reference_urgency}, "
        f"LLM condition={llm_diagnosis}, LLM urgency={llm_urgency}, reward={reward:.2f}. "
        f"Reward features: {', '.join(supporting)}. Reward components: {reward_components}."
    )
    if under_escalated:
        rationale += " Severe under-escalation penalty applied."
    if latent:
        rationale += " Latent risks: " + ", ".join(f"{name}:{score:.2f}" for name, score in latent.items())

    return RewardBreakdown(
        predicted_condition=llm_diagnosis,
        reference_condition=reference_condition,
        urgency=llm_urgency,
        reference_urgency=reference_urgency,
        reward=reward,
        reward_components=reward_components,
        under_escalated=under_escalated,
        rationale=rationale,
        latent_risks=latent,
        supporting_features=supporting,
    )
