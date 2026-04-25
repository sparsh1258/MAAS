# MAAS Submission Evidence

This file collects the main artifacts a reviewer needs for the OpenEnv hackathon submission.

## Training Scripts

- Primary multi-step PPO loop: [`../train_openenv_ppo.py`](../train_openenv_ppo.py)
- Colab-friendly PPO entrypoint: [`../train_trl.py`](../train_trl.py)
- Optional GRPO / Unsloth path: [`../train_grpo.py`](../train_grpo.py)
- PPO notebook: [`../niva_training.ipynb`](../niva_training.ipynb)
- GRPO notebook: [`../niva_grpo_training.ipynb`](../niva_grpo_training.ipynb)

## Environment + Manifest

- OpenEnv environment: [`../environment.py`](../environment.py)
- Reward model: [`../xai_reward_model.py`](../xai_reward_model.py)
- Manifest: [`../openenv.yaml`](../openenv.yaml)
- Space deployment files: [`../Dockerfile`](../Dockerfile), [`../requirements-space.txt`](../requirements-space.txt), [`../.dockerignore`](../.dockerignore)

## Checked-in Results

- Training curve: [`maas_deep_policy_demo/training_curve.png`](maas_deep_policy_demo/training_curve.png)
- Training history: [`maas_deep_policy_demo/training_history.json`](maas_deep_policy_demo/training_history.json)
- Demo summary: [`maas_deep_policy_demo/demo_summary.json`](maas_deep_policy_demo/demo_summary.json)
- Baseline report: [`baseline_report.md`](baseline_report.md)
- Baseline vs trained summary: [`baseline_vs_trained.json`](baseline_vs_trained.json)

## Hugging Face Artifacts

- HF repo mirror / code snapshot: [sparsh122/MAAS](https://huggingface.co/sparsh122/MAAS)
- Latest post-fix GRPO artifacts: [sparsh122/maas-grpo-qwen05b-fix2](https://huggingface.co/sparsh122/maas-grpo-qwen05b-fix2)
- Earlier post-fix GRPO artifacts: [sparsh122/maas-grpo-qwen05b-fix1](https://huggingface.co/sparsh122/maas-grpo-qwen05b-fix1)
- Earlier pre-fix GRPO artifacts: [sparsh122/maas-grpo-qwen05b](https://huggingface.co/sparsh122/maas-grpo-qwen05b)

## Key Metrics

From `maas_deep_policy_demo/demo_summary.json`:

- validation condition accuracy: `0.9792`
- validation urgency accuracy: `1.0000`
- validation loss: `0.0808`

From `baseline_report.md`:

- baseline benchmark score: `0.3367`
- PPO loop reached rollout / reward logging stage on the checked-in small-model run

From the latest post-fix HF GRPO run (`69ed2261d70108f37acdef0e`):

- job created on April 26, 2026 at 01:51 IST and completed successfully
- uploaded `model.safetensors`, `training_summary.json`, and completion parquet logs to `sparsh122/maas-grpo-qwen05b-fix2`
- `train_loss`: `0.0093`
- only one clearly non-flat step repeated the earlier pattern:
  `grad_norm = 10.125`, `reward_std = 5.8513`, `rewards/reward_fn/mean = -7.9375`
- most steps still had `grad_norm = 0`, `reward_std = 0`, and `rewards/reward_fn/mean` near `-3.8`
- mean benchmark score remained about `0.01`

## Latest HF Status

- The earlier GRPO jobs on April 26, 2026 at 00:31 IST and 00:44 IST completed operationally but flatlined with `reward = -20`, `grad_norm = 0`, and `train_loss = 0`.
- The first post-fix run `fix1` proved the hard reward collapse was fixed because it produced a non-zero-gradient step and non-zero reward variance.
- The latest 3-epoch run `fix2` confirms the HF pipeline is healthy and uploads reliably, but meaningful task improvement is still not proven.
- The best training next step is to inspect the uploaded completions and adjust prompt / reward / task diversity before launching another GRPO rerun.

## Space Note

- `sparsh122/MAAS` is a Hugging Face model repo mirror, not a Space URL.
- The live demo should only be promoted as the primary public Space once the actual Space app is synced and booting.

## What Changed in the Environment

- Partial observability with withheld signals
- Multi-step actions: `assess`, `request_signal`, `diagnose`
- Temporal belief-state updates across visible check-in days
- Class-weighted safety reward with explicit `reward_components`
- Looped inference path in [`../inference.py`](../inference.py)

## Presentation Material

- Slide deck: [OpenEnv Hackathon Deck](https://docs.google.com/presentation/d/1KzV0MxZYYA6PXXJ-nAcSRUn5staJkfQvEgHF1QVl5as/preview?pru=AAABnedodns*3ITAIB6zwg6GBoSPLOY7LQ&slide=id.g3e610e50443_9_233)

## Judge-Facing Interpretation

- The strongest submission evidence today is the environment design itself: professional workflow structure, temporal state, hidden information, and deterministic safety-grounded reward shaping.
- The checked-in PPO artifacts already show that MAAS is not just a static app; it has an actual trainable loop with recorded metrics and benchmark summaries.
- The post-fix HF GRPO runs are useful evidence of cloud-training readiness, but they should be framed as an improving training pipeline rather than a final breakthrough result.
