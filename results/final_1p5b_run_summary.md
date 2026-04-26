# Final 1.5B GRPO Run Summary

Run date: April 26, 2026

Job:
- Hugging Face Job ID: `69ed7e0dd2c8bd8662bcef36`
- Hardware: `a10g-small`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Training config: `epochs=2`, `batch_size=1`, `gradient_accumulation_steps=4`, `num_generations=4`

Outcome:
- The GRPO training phase completed all `18/18` steps and wrote model shards successfully.
- The final upload to the Hub failed because the token had read access but not LFS write permission.
- This means the training evidence is valid, but the final model artifact was not persisted to the Hub from this run.

Key takeaways:
- Reward variance was alive for almost the entire run, which means the GRPO setup was not collapsed.
- Only one visible step showed `reward_std = 0`, and that step also had `grad_norm = 0`, indicating a temporary degenerate batch rather than a totally dead run.
- The best observed batch reached:
  - `mean_reward = 3.306`
  - `reward_std = 15.06`
  - `mean_benchmark_score = 0.255`
  - `mean_safety_reward = 1.019`
  - `exact_json_rate = 0.75`
- Final logged batch metrics were:
  - `mean_reward = -2.612`
  - `reward_std = 2.375`
  - `mean_benchmark_score = 0.0575`
  - `mean_safety_reward = -0.95`
  - `exact_json_rate = 0.75`

Whole-run training stats:
- `train_runtime = 96.62s`
- `train_steps_per_second = 0.186`
- `train_loss = 0.06307`

Interpretation for submission:
- The important proof point is that this was a real online RL run with non-zero gradients, non-zero reward variance, and meaningful reward swings across sampled completions.
- This run is stronger evidence than the earlier broken jobs because it demonstrates the GRPO loop operating end to end on cloud GPU infrastructure.
- However, the final benchmark signal is still noisy, so this should be presented as a working RL training pipeline with live reward shaping rather than as a conclusively superior final clinical model.

Files produced from the completed logs:
- `results/final_1p5b_run_metrics.csv`
- `results/final_1p5b_reward_chart.svg`
- `results/final_1p5b_quality_chart.svg`
