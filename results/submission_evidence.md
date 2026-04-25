# MAAS Submission Evidence

This file collects the main artifacts a reviewer needs for the OpenEnv hackathon submission.

## Training Scripts

- Primary multi-step PPO loop: [`../train_openenv_ppo.py`](../train_openenv_ppo.py)
- Colab-friendly PPO entrypoint: [`../train_trl.py`](../train_trl.py)
- Optional GRPO / Unsloth path: [`../train_grpo.py`](../train_grpo.py)
- Notebook: [`../niva_training.ipynb`](../niva_training.ipynb)

## Environment + Manifest

- OpenEnv environment: [`../environment.py`](../environment.py)
- Reward model: [`../xai_reward_model.py`](../xai_reward_model.py)
- Manifest: [`../openenv.yaml`](../openenv.yaml)

## Checked-in Results

- Training curve: [`maas_deep_policy_demo/training_curve.png`](maas_deep_policy_demo/training_curve.png)
- Training history: [`maas_deep_policy_demo/training_history.json`](maas_deep_policy_demo/training_history.json)
- Demo summary: [`maas_deep_policy_demo/demo_summary.json`](maas_deep_policy_demo/demo_summary.json)
- Baseline report: [`baseline_report.md`](baseline_report.md)
- Baseline vs trained summary: [`baseline_vs_trained.json`](baseline_vs_trained.json)

## Key Metrics

From `maas_deep_policy_demo/demo_summary.json`:

- validation condition accuracy: `0.9792`
- validation urgency accuracy: `1.0000`
- validation loss: `0.0808`

From `baseline_report.md`:

- baseline benchmark score: `0.3367`
- PPO loop reached rollout / reward logging stage on the checked-in small-model run

## What Changed in the Environment

- Partial observability with withheld signals
- Multi-step actions: `assess`, `request_signal`, `diagnose`
- Temporal belief-state updates across visible check-in days
- Class-weighted safety reward with explicit `reward_components`
- Looped inference path in [`../inference.py`](../inference.py)

## Presentation Material

- Slide deck: [OpenEnv Hackathon Deck](https://docs.google.com/presentation/d/1KzV0MxZYYA6PXXJ-nAcSRUn5staJkfQvEgHF1QVl5as/preview?pru=AAABnedodns*3ITAIB6zwg6GBoSPLOY7LQ&slide=id.g3e610e50443_9_233)
