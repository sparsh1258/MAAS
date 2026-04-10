"""
RL-style risk policy for Niva.

This module keeps the OpenEnv action space stable while replacing the simple
if/else classifier with a compact linear value-function policy. The weights are
seeded from maternal-health heuristics and can be updated from rewards, which
makes the environment behave like an RL benchmark without requiring a large
training job during hackathon validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


SAFE_CONDITIONS = [
    "preeclampsia",
    "gestational_diabetes",
    "anemia",
    "preterm_risk",
    "fetal_distress",
    "low_risk",
]

LATENT_RISKS = {
    "postpartum_hemorrhage": "bleeding and weakness pattern",
    "maternal_infection": "pain, dizziness, low sleep, systemic strain",
    "dehydration": "low water and dizziness pattern",
    "intrahepatic_cholestasis": "late-pregnancy discomfort proxy signal",
    "placental_abruption": "bleeding plus abdominal pain proxy signal",
    "maternal_exhaustion": "low sleep, low energy, high breathlessness",
    "nutrition_deficit": "low meals and low energy trend",
}

URGENCY_ORDER = ["monitor_at_home", "visit_phc_this_week", "go_to_hospital_today"]


@dataclass(frozen=True)
class PolicyResult:
    condition: str
    urgency: str
    q_values: Dict[str, float]
    confidence: float
    latent_risks: Dict[str, float]
    rationale: str


class RLMaternalRiskPolicy:
    """Small approximate Q-value policy for maternal triage decisions."""

    def __init__(self, learning_rate: float = 0.04, discount: float = 0.92):
        self.learning_rate = learning_rate
        self.discount = discount
        self.weights = self._seed_weights()

    def _seed_weights(self) -> Dict[str, Dict[str, float]]:
        return {
            "preeclampsia": {
                "danger_bp": 7.0,
                "high_bp": 4.0,
                "bp_rising": 2.5,
                "headache_swelling": 3.8,
                "vision_headache": 4.2,
                "history_htn": 1.6,
                "history_preeclampsia": 2.6,
                "trimester3": 0.8,
                "poor_sleep": 0.4,
            },
            "gestational_diabetes": {
                "history_diabetes": 4.3,
                "high_meals": 1.4,
                "low_energy": 1.2,
                "breathlessness": 1.0,
                "trimester2_or_3": 0.7,
                "weight_signal": 0.6,
            },
            "anemia": {
                "low_nutrition": 4.0,
                "very_low_meals": 2.4,
                "low_energy": 2.2,
                "breathlessness": 2.4,
                "dizziness": 1.8,
                "poor_sleep": 0.8,
            },
            "preterm_risk": {
                "bleeding": 6.0,
                "abdominal_pain": 3.0,
                "history_complication": 3.2,
                "weeks_early_third": 1.8,
                "poor_sleep": 1.0,
                "maternal_strain": 0.8,
            },
            "fetal_distress": {
                "danger_low_kicks": 7.5,
                "low_kick_avg": 5.0,
                "late_pregnancy": 1.2,
                "maternal_strain": 0.7,
            },
            "low_risk": {
                "normal_bp": 1.8,
                "good_kicks": 2.2,
                "good_meals": 1.2,
                "good_sleep": 1.0,
                "no_history": 1.6,
                "no_flags": 2.6,
            },
        }

    def featurize(self, obs) -> Dict[str, float]:
        flags = set(obs.risk_flags or [])
        history = set(obs.history_flags or [])
        avg_meals = obs.avg_meals or 0
        avg_sleep = obs.avg_sleep or 0
        kicks = obs.avg_kick_count
        energy = obs.latest_energy or 0
        breath = obs.latest_breathlessness or 0
        weeks = obs.weeks_pregnant or 0

        features = {
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
        return features

    def q_values(self, features: Dict[str, float]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for condition, weights in self.weights.items():
            values[condition] = sum(features.get(name, 0.0) * weight for name, weight in weights.items())
        return values

    def latent_risk_scores(self, features: Dict[str, float]) -> Dict[str, float]:
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

    def choose_urgency(self, condition: str, features: Dict[str, float], confidence: float) -> str:
        if features["danger_bp"] or features["danger_low_kicks"] or features["bleeding"] or features["vision_headache"]:
            return "go_to_hospital_today"
        if condition in {"preeclampsia", "fetal_distress"} and confidence >= 0.45:
            return "go_to_hospital_today"
        if condition != "low_risk":
            return "visit_phc_this_week"
        return "monitor_at_home"

    def predict(self, obs) -> PolicyResult:
        features = self.featurize(obs)
        q_values = self.q_values(features)
        ranked = sorted(q_values.items(), key=lambda item: item[1], reverse=True)
        condition, top_q = ranked[0]
        second_q = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_q < 1.7:
            condition = "low_risk"
            top_q = q_values.get("low_risk", 0.0)
        confidence = round(min(0.99, max(0.05, (top_q - second_q + 1.0) / 8.0)), 3)
        urgency = self.choose_urgency(condition, features, confidence)
        latent = self.latent_risk_scores(features)
        strongest_features = [name for name, value in sorted(features.items(), key=lambda item: item[1], reverse=True) if value][:5]
        rationale = (
            f"RL policy selected {condition.replace('_', ' ')} with Q={top_q:.2f}, "
            f"confidence={confidence:.2f}, urgency={urgency}. "
            f"Top factors: {', '.join(strongest_features) or 'baseline healthy pattern'}."
        )
        if latent:
            latent_names = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(latent.items(), key=lambda item: item[1], reverse=True)[:3])
            rationale += f" Latent risks monitored: {latent_names}."
        return PolicyResult(condition, urgency, q_values, confidence, latent, rationale)

    def update_from_reward(self, obs, predicted: str, reward: float) -> None:
        if predicted not in self.weights:
            return
        features = self.featurize(obs)
        prediction = self.q_values(features).get(predicted, 0.0)
        td_error = reward - prediction
        for name, value in features.items():
            if value:
                self.weights[predicted][name] = self.weights[predicted].get(name, 0.0) + self.learning_rate * td_error * value


RL_RISK_MODEL = RLMaternalRiskPolicy()
