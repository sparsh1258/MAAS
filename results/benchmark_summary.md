# MAAS Benchmark Summary

## One-Line Summary

MAAS is a multi-turn maternal-health OpenEnv benchmark in which an agent must triage prenatal risk over a three-day partially observable trajectory while prioritizing safe escalation.

## What Is Being Benchmarked

- **Task type:** professional maternal triage workflow
- **Setting:** low-resource prenatal monitoring with delayed / hidden evidence
- **Loop:** `reset -> multi-turn action -> env.step -> updated state -> final diagnosis`
- **Actions:** `request_bp_recheck`, `request_kick_count`, `advance_day`, `refer_to_phc`, `diagnose`
- **Objective:** maximize condition + urgency correctness while minimizing unsafe under-escalation

## Why It Is OpenEnv-Relevant

- The benchmark is **stateful**, not one-shot.
- The agent interacts with a **partially observable world**.
- Useful signals are intentionally **revealed over time**.
- Reward depends on **workflow behavior**, not just label prediction.

## Current Evidence In Repo

### Environment Evidence

- Multi-turn manifest: [`../openenv.yaml`](../openenv.yaml)
- Environment logic: [`../environment.py`](../environment.py)
- API runtime: [`../main.py`](../main.py)
- Judge-style inference runner: [`../inference.py`](../inference.py)

### Training Evidence

- Reward curve: [`grpo_reward_curve.png`](grpo_reward_curve.png)
- Loss curve: [`grpo_loss_curve.png`](grpo_loss_curve.png)
- Demo training curve: [`maas_deep_policy_demo/training_curve.png`](maas_deep_policy_demo/training_curve.png)
- GRPO summary: [`grpo_training_summary.json`](grpo_training_summary.json)
- 1.5B run summary: [`final_1p5b_run_summary.md`](final_1p5b_run_summary.md)

## Key Metrics Already Checked In

From `maas_deep_policy_demo/demo_summary.json`:

- validation condition accuracy: `0.9792`
- validation urgency accuracy: `1.0000`
- validation loss: `0.0808`

From `baseline_report.md`:

- average baseline score: `0.3367`
- PPO loop initialized and emitted reward logs before a later CUDA generation failure

From `final_1p5b_run_summary.md`:

- full `18/18` GRPO steps completed
- non-zero reward variance for most of the run
- best observed batch:
  - `mean_reward = 3.306`
  - `reward_std = 15.06`
  - `mean_benchmark_score = 0.255`
  - `exact_json_rate = 0.75`

## Honest Interpretation

What is clearly demonstrated:

- the MAAS benchmark is a real multi-turn OpenEnv task
- reward shaping is connected to maternal-health safety behavior
- the training pipeline runs end to end
- the submission includes checked-in plots and summaries

What remains future work:

- final policy quality is still noisy
- the strongest claim is **benchmark design + trainable loop + deployment readiness**, not finished clinical performance

## Recommended Judge Framing

Judge MAAS as:

1. a maternal-health professional workflow benchmark
2. a multi-turn partially observable OpenEnv environment
3. a safety-first RL packaging effort with reproducible evidence
