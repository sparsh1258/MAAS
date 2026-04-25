# Baseline and Training Report

## Baseline
- Baseline inference was executed successfully.
- Output file: `results/baseline_output.txt`
- Average baseline score observed: `0.3367`

## Training Attempt
- Training script used: `train_openenv_ppo.py`
- Model used: `Qwen/Qwen2-0.5B-Instruct`
- PPO loop initialized successfully.
- Model and tokenizer loaded successfully.
- Rollout started and emitted reward logs.
- Observed reward log: `mean_reward = -12.0`

## Outcome
- The RL environment, verifier, and PPO training stack were connected successfully.
- Training progressed into rollout/logging stage.
- The run stopped later with a CUDA generation error.

## Why this still matters
This demonstrates that the project is not just a static app. It includes a real environment -> model -> reward -> PPO training loop, and the model was actively run inside that environment.
