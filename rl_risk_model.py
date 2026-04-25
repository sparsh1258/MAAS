"""
Backward-compatible shim.

The learnable RL policy has been retired. Import deterministic safety / reward
logic from xai_reward_model instead.
"""

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
