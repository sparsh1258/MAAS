"""
Production-grade heuristic+learnable policy for MAAS.

The environment is deterministic and safety-sensitive; this policy provides:
- 40+ engineered features from the Observation.
- Condition-specific linear weights with clinical heuristics.
- Softmax normalization to produce confidence.
- Trimester amplifier for danger conditions.
- Ensemble tie-breaker for close calls.
- A small prioritized replay buffer with TD-like weight updates.

Constraints: stdlib only (plus existing project deps).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from xai_reward_model import (
    SAFE_CONDITIONS,
    URGENCY_ORDER,
    calculate_reward,
    choose_urgency,
    featurize,
    infer_reference_condition,
    latent_risk_scores,
    supporting_features,
)


DANGER_CONDITIONS = {"preeclampsia", "fetal_distress", "preterm_risk"}


@dataclass
class PolicyResult:
    condition: str
    urgency: str
    q_values: Dict[str, float]
    confidence: float
    latent_risks: Dict[str, float]
    rationale: str


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    s = sum(exps.values()) or 1.0
    return {k: (v / s) for k, v in exps.items()}


def _access_penalty(access_tier: str) -> float:
    # As requested: rural=0.3, semi-urban=0.6, urban=1.0
    t = (access_tier or "").lower()
    if "urban" in t:
        return 1.0
    if "rural" in t or "tribal" in t:
        return 0.3
    return 0.6


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class RLMaternalRiskPolicy:
    def __init__(self) -> None:
        self.feature_names: List[str] = self._feature_list()
        self.weights: Dict[str, Dict[str, float]] = self._init_weights()
        self.bias: Dict[str, float] = {c: 0.0 for c in SAFE_CONDITIONS}

        # Prioritized replay: store high-|reward| experiences more often.
        self.replay: deque[Tuple[Dict[str, float], str, float, str]] = deque(maxlen=500)
        self.lr: float = 0.03
        self.gamma: float = 0.92
        self._updates: int = 0

    def _feature_list(self) -> List[str]:
        # 40+ features. Some come from xai_reward_model.featurize; others are new.
        return [
            # Existing core features (from featurize)
            "danger_bp",
            "high_bp",
            "bp_rising",
            "headache_swelling",
            "vision_headache",
            "danger_low_kicks",
            "low_kick_avg",
            "bleeding",
            "abdominal_pain",
            "low_nutrition",
            "very_low_meals",
            "high_meals",
            "good_meals",
            "poor_sleep",
            "good_sleep",
            "low_energy",
            "breathlessness",
            "maternal_strain",
            "history_diabetes",
            "history_htn",
            "history_preeclampsia",
            "history_complication",
            "no_history",
            "trimester3",
            "trimester2_or_3",
            "weeks_early_third",
            "late_pregnancy",
            "weight_signal",
            "normal_bp",
            "good_kicks",
            "no_flags",
            "dizziness",
            # New engineered features
            "systolic_bp_normalized",
            "diastolic_bp_normalized",
            "kick_count_normalized",
            "trimester_week_combo",
            "composite_danger_score",
            "symptom_cluster_count",
            "history_risk_score",
            "nutrition_sleep_deficit",
            "maternal_strain_index",
            "regional_access_penalty",
            "data_confidence",
            "composite_risk_score",
            "severe_anemia_proxy",
            "maternal_exhaustion_proxy",
            "preterm_contraction_proxy",
            "vomiting_signal",
            "rapid_weight_gain_proxy",
        ]

    def _init_weights(self) -> Dict[str, Dict[str, float]]:
        # Clinical heuristic seeds. Danger conditions get large weights on danger features.
        w: Dict[str, Dict[str, float]] = {c: {f: 0.0 for f in self.feature_names} for c in SAFE_CONDITIONS}

        def setw(cond: str, feat: str, val: float) -> None:
            if cond in w and feat in w[cond]:
                w[cond][feat] = float(val)

        # Preeclampsia
        for f, v in [
            ("danger_bp", 9.0),
            ("high_bp", 6.0),
            ("bp_rising", 4.0),
            ("headache_swelling", 5.5),
            ("vision_headache", 7.5),
            ("rapid_weight_gain_proxy", 3.5),
            ("symptom_cluster_count", 1.4),
            ("composite_danger_score", 3.2),
            ("composite_risk_score", 2.0),
            ("history_htn", 2.5),
            ("history_preeclampsia", 3.0),
            ("regional_access_penalty", 0.6),
        ]:
            setw("preeclampsia", f, v)

        # Fetal distress
        for f, v in [
            ("danger_low_kicks", 9.0),
            ("low_kick_avg", 6.5),
            ("late_pregnancy", 3.5),
            ("maternal_strain", 2.0),
            ("maternal_strain_index", 2.4),
            ("composite_danger_score", 3.0),
            ("symptom_cluster_count", 0.6),
        ]:
            setw("fetal_distress", f, v)

        # Preterm risk
        for f, v in [
            ("bleeding", 7.0),
            ("abdominal_pain", 4.5),
            ("preterm_contraction_proxy", 6.0),
            ("weeks_early_third", 2.0),
            ("history_complication", 2.8),
            ("poor_sleep", 1.5),
            ("maternal_exhaustion_proxy", 1.8),
            ("composite_danger_score", 2.4),
        ]:
            setw("preterm_risk", f, v)

        # Gestational diabetes
        for f, v in [
            ("history_diabetes", 4.0),
            ("high_meals", 1.7),
            ("weight_signal", 2.8),
            ("trimester2_or_3", 1.6),
            ("low_energy", 1.2),
            ("nutrition_sleep_deficit", 0.8),
        ]:
            setw("gestational_diabetes", f, v)

        # Anemia
        for f, v in [
            ("low_nutrition", 3.6),
            ("very_low_meals", 4.2),
            ("dizziness", 2.6),
            ("breathlessness", 2.4),
            ("low_energy", 2.8),
            ("severe_anemia_proxy", 7.2),
            ("maternal_exhaustion_proxy", 1.6),
            ("nutrition_sleep_deficit", 1.4),
        ]:
            setw("anemia", f, v)

        # Low risk: reward absence of danger and good lifestyle signals.
        for f, v in [
            ("no_flags", 3.2),
            ("normal_bp", 2.5),
            ("good_kicks", 2.0),
            ("good_meals", 1.7),
            ("good_sleep", 1.6),
            ("data_confidence", 1.2),
            ("composite_risk_score", -3.0),
            ("composite_danger_score", -4.0),
        ]:
            setw("low_risk", f, v)

        # Small negative cross-penalties (help separability).
        for cond in SAFE_CONDITIONS:
            if cond != "low_risk":
                setw(cond, "no_flags", -1.0)
                setw(cond, "good_sleep", -0.3)
                setw(cond, "good_meals", -0.3)
        setw("low_risk", "danger_bp", -7.0)
        setw("low_risk", "danger_low_kicks", -7.0)
        setw("low_risk", "bleeding", -7.0)

        return w

    def _engineer_features(self, obs: Any) -> Dict[str, float]:
        base = featurize(obs)
        flags = set(getattr(obs, "risk_flags", []) or [])
        symptoms = list(getattr(obs, "symptom_cluster", []) or [])
        history = set(getattr(obs, "history_flags", []) or [])

        sys_bp = getattr(obs, "bp_systolic_latest", None)
        dia_bp = getattr(obs, "bp_diastolic_latest", None)
        weeks = _safe_float(getattr(obs, "weeks_pregnant", 0), 0.0)
        trimester = _safe_float(getattr(obs, "trimester", 1), 1.0)
        avg_kicks = getattr(obs, "avg_kick_count", None)
        avg_meals = _safe_float(getattr(obs, "avg_meals", 0.0), 0.0)
        avg_sleep = _safe_float(getattr(obs, "avg_sleep", 0.0), 0.0)
        energy = _safe_float(getattr(obs, "latest_energy", 0.0), 0.0)
        breath = _safe_float(getattr(obs, "latest_breathlessness", 0.0), 0.0)
        days = _safe_float(getattr(obs, "days_of_data", 0), 0.0)

        # Requested engineered features.
        sys_norm = 0.0
        dia_norm = 0.0
        if sys_bp is not None:
            sys_norm = _clip01((_safe_float(sys_bp) - 90.0) / 110.0)  # 90..200
        else:
            sys_norm = 1.0 if "DANGER_BP_CRITICAL" in flags else 0.6 if "HIGH_BP" in flags else 0.3
        if dia_bp is not None:
            dia_norm = _clip01((_safe_float(dia_bp) - 50.0) / 90.0)  # 50..140
        else:
            dia_norm = 1.0 if "DANGER_BP_CRITICAL" in flags else 0.6 if "HIGH_BP" in flags else 0.3

        kick_norm = 0.0
        if avg_kicks is not None:
            kick_norm = _clip01(_safe_float(avg_kicks) / 10.0)
        else:
            kick_norm = 0.1 if "DANGER_LOW_KICKS" in flags else 0.6

        trimester_week_combo = _clip01((weeks / 42.0) * (trimester / 3.0))
        symptom_cluster_count = float(len(symptoms))

        # Composite danger score: weighted sum of danger flags and proxies.
        danger_score = 0.0
        for f in flags:
            if f.startswith("DANGER_"):
                danger_score += 1.0
            elif f.startswith("HIGH_") or f.startswith("WARN_"):
                danger_score += 0.5
            elif f.endswith("_PROXY") or f.endswith("_HIGH") or f.endswith("_SIGNAL"):
                danger_score += 0.6
        danger_score = _clip01(danger_score / 5.0)

        # History risk score.
        history_risk = 0.0
        for h in history:
            if "preeclampsia" in h or "htn" in h or "hypertension" in h:
                history_risk += 0.8
            elif "complication" in h or "prev_" in h:
                history_risk += 0.6
            elif "diabetes" in h or "gdm" in h:
                history_risk += 0.5
            else:
                history_risk += 0.2
        history_risk_score = _clip01(history_risk / 2.2)

        nutrition_sleep_deficit = _clip01(max(0.0, (2.8 - avg_meals) / 2.3) * 0.55 + max(0.0, (6.5 - avg_sleep) / 6.5) * 0.45)
        maternal_strain_index = _clip01(max(0.0, (6.0 - energy) / 6.0) * 0.55 + max(0.0, (breath - 4.0) / 6.0) * 0.45)
        access_pen = _access_penalty(getattr(obs, "regional_access_tier", "semi_urban"))
        data_confidence = _clip01(days / 3.0)
        composite_risk_score = _clip01(_safe_float(getattr(obs, "composite_risk_score", 0.0), 0.0))

        engineered = dict(base)
        engineered.update(
            {
                "systolic_bp_normalized": sys_norm,
                "diastolic_bp_normalized": dia_norm,
                "kick_count_normalized": kick_norm,
                "trimester_week_combo": trimester_week_combo,
                "composite_danger_score": danger_score,
                "symptom_cluster_count": symptom_cluster_count,
                "history_risk_score": history_risk_score,
                "nutrition_sleep_deficit": nutrition_sleep_deficit,
                "maternal_strain_index": maternal_strain_index,
                "regional_access_penalty": access_pen,
                "data_confidence": data_confidence,
                "composite_risk_score": composite_risk_score,
                "severe_anemia_proxy": float("SEVERE_ANEMIA_PROXY" in flags),
                "maternal_exhaustion_proxy": float("MATERNAL_EXHAUSTION" in flags),
                "preterm_contraction_proxy": float("PRETERM_CONTRACTION_PROXY" in flags),
                "vomiting_signal": float("VOMITING_SIGNAL" in flags),
                "rapid_weight_gain_proxy": float("RAPID_WEIGHT_GAIN" in flags),
            }
        )
        # Ensure all requested keys exist.
        for f in self.feature_names:
            engineered.setdefault(f, 0.0)
        return engineered

    def _score_condition(self, features: Dict[str, float], condition: str) -> float:
        s = self.bias.get(condition, 0.0)
        w = self.weights.get(condition, {})
        for fname in self.feature_names:
            s += w.get(fname, 0.0) * float(features.get(fname, 0.0))
        # Trimester amplifier for danger conditions.
        trimester = features.get("trimester3", 0.0) * 3.0 + (1.0 - features.get("trimester3", 0.0)) * (2.0 if features.get("trimester2_or_3", 0.0) else 1.0)
        if condition in DANGER_CONDITIONS:
            s *= 1.0 + 0.15 * (trimester - 1.0)
        return float(s)

    def _tie_break(self, features: Dict[str, float], top: str, runner_up: str) -> str:
        """
        Secondary rule-based check when top two scores are close.
        """
        flags_danger = features.get("danger_bp", 0.0) + features.get("danger_low_kicks", 0.0) + features.get("bleeding", 0.0) + features.get("vision_headache", 0.0)
        if flags_danger >= 1.0:
            # Prefer a danger condition if danger flags are active.
            if features.get("danger_bp", 0.0) or features.get("vision_headache", 0.0) or features.get("headache_swelling", 0.0):
                return "preeclampsia"
            if features.get("danger_low_kicks", 0.0) or features.get("low_kick_avg", 0.0):
                return "fetal_distress"
            if features.get("bleeding", 0.0) or features.get("abdominal_pain", 0.0) or features.get("preterm_contraction_proxy", 0.0):
                return "preterm_risk"
        # If severe anemia proxy triggers, break toward anemia.
        if features.get("severe_anemia_proxy", 0.0) >= 1.0:
            return "anemia"
        return top

    def predict(self, obs: Any) -> PolicyResult:
        feats = self._engineer_features(obs)

        scores: Dict[str, float] = {c: self._score_condition(feats, c) for c in SAFE_CONDITIONS}
        probs = _softmax(scores)

        # Select top-2
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_cond = ordered[0][0]
        runner = ordered[1][0] if len(ordered) > 1 else top_cond
        if ordered and len(ordered) > 1 and (ordered[0][1] - ordered[1][1]) <= 1.5:
            top_cond = self._tie_break(feats, top_cond, runner)

        # Urgency: start from deterministic chooser, then adjust for access.
        # Use xai features as a safety backstop.
        urgency = choose_urgency(top_cond, featurize(obs))
        # In very low-access contexts, prefer earlier escalation when danger is plausible.
        access = feats["regional_access_penalty"]
        if access <= 0.35 and top_cond in DANGER_CONDITIONS and urgency != "go_to_hospital_today":
            if feats["composite_danger_score"] >= 0.35 or feats["composite_risk_score"] >= 0.35:
                urgency = "go_to_hospital_today"

        latent = latent_risk_scores(featurize(obs))

        confidence = float(probs.get(top_cond, 0.0))
        confidence = max(0.01, min(0.99, confidence))

        try:
            support = supporting_features(infer_reference_condition(obs), featurize(obs))
            rationale = " | ".join(str(s) for s in (support or [])[:6])
        except Exception:
            rationale = "Policy selected safest triage given engineered features."

        return PolicyResult(
            condition=top_cond,
            urgency=urgency,
            q_values=scores,
            confidence=confidence,
            latent_risks=latent if isinstance(latent, dict) else {},
            rationale=rationale,
        )

    def update_from_reward(self, obs: Any, predicted: str, reward: float) -> None:
        """
        TD-like update on predicted condition + contrastive update on runner-up.

        Stores experience and replays 4 samples for stability.
        """
        try:
            feats = self._engineer_features(obs)
        except Exception:
            return

        pred = predicted if predicted in SAFE_CONDITIONS else infer_reference_condition(obs)
        r = float(reward)

        # Compute current scores and runner-up for contrastive learning.
        scores = {c: self._score_condition(feats, c) for c in SAFE_CONDITIONS}
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ordered[0][0]
        runner = ordered[1][0] if len(ordered) > 1 else top

        # Prioritize by absolute reward magnitude.
        self.replay.append((feats, pred, r, runner))

        def td_update(fv: Dict[str, float], a: str, rew: float, runner_up: str) -> None:
            q = self._score_condition(fv, a)
            target = rew + self.gamma * max(self._score_condition(fv, c) for c in SAFE_CONDITIONS)
            err = target - q

            # Update weights for selected action.
            for name in self.feature_names:
                self.weights[a][name] += self.lr * err * float(fv.get(name, 0.0))
            self.bias[a] += self.lr * err

            # Contrastive: small negative update to runner-up to increase margin.
            if runner_up in SAFE_CONDITIONS and runner_up != a:
                for name in self.feature_names:
                    self.weights[runner_up][name] -= (self.lr * 0.15) * err * float(fv.get(name, 0.0))
                self.bias[runner_up] -= (self.lr * 0.15) * err

        td_update(feats, pred, r, runner)

        # Replay 4 past experiences (prioritized by |reward|, approximated by sampling bias).
        if len(self.replay) >= 8:
            batch = sorted(list(self.replay)[-40:], key=lambda t: abs(t[2]), reverse=True)[:10]
            for _ in range(4):
                fv, a, rew, ru = batch[int(math.floor(abs(math.sin(self._updates + 1)) * (len(batch) - 1)))]
                td_update(fv, a, rew, ru)

        self._updates += 1


RL_RISK_MODEL = RLMaternalRiskPolicy()
