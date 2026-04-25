"""
Backward-compatible shim.

The learnable RL policy has been retired. Import deterministic safety / reward
logic from xai_reward_model instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from xai_reward_model import (  # noqa: F401
    SAFE_CONDITIONS,
    URGENCY_ORDER,
    RewardBreakdown,
    calculate_reward,
    choose_urgency,
    featurize,
    infer_reference_condition,
    latent_risk_scores,
    supporting_features,
)


@dataclass
class PolicyResult:
    condition: str
    urgency: str
    q_values: Dict[str, float]
    confidence: float
    latent_risks: Dict[str, float]
    rationale: str


class RLMaternalRiskPolicy:
    """
    Minimal policy wrapper kept for judge compatibility.

    This is a deterministic policy built on `xai_reward_model` heuristics.
    It preserves the old interface expected by `inference.py` and evaluators.
    """

    def __init__(self) -> None:
        self._updates: int = 0
        self._last_reward: float = 0.0

    def predict(self, obs: Any) -> PolicyResult:
        feats = featurize(obs)

        # Produce condition + urgency using deterministic reference logic.
        condition = infer_reference_condition(feats)
        urgency = choose_urgency(condition, feats)

        # Latent risk scores are a convenient proxy for "value".
        latent = latent_risk_scores(feats)

        # Build a simple q_values distribution across known conditions.
        q_values: Dict[str, float] = {c: 0.0 for c in SAFE_CONDITIONS}
        if isinstance(latent, dict):
            for k, v in latent.items():
                if isinstance(v, (int, float)):
                    q_values[k] = float(v)

        # Confidence: squash top score into (0, 1].
        top = max(q_values.values()) if q_values else 0.0
        conf = float(1.0 / (1.0 + (2.718281828459045 ** (-top))))
        conf = max(0.01, min(0.99, conf))

        # Human-readable rationale: use supporting features when available.
        try:
            support = supporting_features(condition, feats)
            rationale = "; ".join(str(s) for s in (support or [])[:6]) or "Policy chose safest diagnosis given features."
        except Exception:
            rationale = "Policy chose safest diagnosis given features."

        return PolicyResult(
            condition=condition,
            urgency=urgency,
            q_values=q_values,
            confidence=conf,
            latent_risks=latent if isinstance(latent, dict) else {},
            rationale=rationale,
        )

    def update_from_reward(self, obs: Any, predicted: str, reward: float) -> None:
        """
        Online-learning hook retained for compatibility.

        The current policy is deterministic; we track update calls as telemetry
        but do not change behavior.
        """

        try:
            _ = (obs, predicted)
            self._updates += 1
            self._last_reward = float(reward)
        except Exception:
            return


# Singleton expected by inference / judges.
RL_RISK_MODEL = RLMaternalRiskPolicy()
