# MAAS Mini Writeup

## Problem

Maternal-health triage in low-resource settings is not a one-shot classification problem. Clinically relevant evidence appears over time, some symptoms are only visible after follow-up, and the cost of unsafe under-escalation is high. MAAS frames this as an OpenEnv workflow task rather than a static prediction benchmark.

## Environment

MAAS is a multi-turn prenatal-health environment with a Gym-style `reset -> step -> state` loop. Each episode unfolds over three days.

- Day 1 exposes only basic vitals and kick count.
- Day 2 reveals symptoms.
- Day 3 reveals history flags plus later-episode context such as meals, sleep, and energy.

The agent must choose among:

- `request_bp_recheck`
- `request_kick_count`
- `advance_day`
- `refer_to_phc`
- `diagnose`

This makes the benchmark partially observable, stateful, and aligned with professional workflow reasoning.

## Safety Framing

The reward is safety-first.

- Danger flags are prioritized.
- Information gathering has a small cost.
- Appropriate PHC referral gets intermediate reward.
- Final diagnosis reward combines condition accuracy, urgency alignment, and safety alignment.
- Under-escalating a danger case is penalized more than asking for another signal.

## Training Evidence

The repo contains:

- runnable GRPO and PPO training scripts
- Colab notebooks for the single-step and multi-turn paths
- checked-in PNG training evidence required by the validator
- stronger 1.5B GRPO run summaries and metrics in `results/final_1p5b_run_summary.md` and `results/final_1p5b_run_metrics.csv`

The current evidence proves that:

- the environment is real
- the RL loop runs end to end
- reward shaping and JSON-structured outputs are wired into training

The current evidence does **not** claim that MAAS is already a clinically production-ready policy. The strongest contribution is the benchmark design, safety-aware reward shaping, deployment packaging, and reproducible training workflow.

## Submission Links

- Hugging Face Space: [sparsh122/maas-openenv](https://huggingface.co/spaces/sparsh122/maas-openenv)
- Slide deck: [OpenEnv Hackathon Deck](https://docs.google.com/presentation/d/1KzV0MxZYYA6PXXJ-nAcSRUn5staJkfQvEgHF1QVl5as/preview?pru=AAABnedodns*3ITAIB6zwg6GBoSPLOY7LQ&slide=id.g3e610e50443_9_233)
- Main README: [`../README.md`](../README.md)
