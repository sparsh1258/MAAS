# MAAS Mini Writeup

MAAS is an OpenEnv-compatible maternal-health triage environment built for
Theme 3.1: professional world-modeling tasks. Instead of a one-step classifier,
the agent works through a three-day partially observable prenatal episode,
decides whether to gather more evidence, and then chooses both a diagnosis and
an urgency tier.

The deployed environment is live on Hugging Face Spaces and exposes real
`/reset`, `/step`, and `/state` endpoints. The key design idea is safety under
uncertainty: requesting more evidence costs a small amount, but missing a
danger case should cost much more. That makes the benchmark more about safe
workflow behavior than label prediction.

The repo includes:

- `MultiTurnPrenatalEnvironment` with temporal state and five actions
- live patient / ASHA and coordinator portals
- GRPO and PPO training scripts
- Colab-ready notebooks
- checked-in GRPO run summaries, reward curves, and training metrics

The strongest current claim is a realistic, safety-sensitive, multi-turn OpenEnv
environment for maternal triage. The weaker claim is model performance: GRPO
infrastructure runs, but robust trained-model improvement still needs exact
multi-turn before/after proof from the same evaluator.

Current deterministic multi-turn evaluation from `evaluation_report.md`:

- `conservative_visible_baseline`: mean reward `0.9551`, condition accuracy
  `1.0000`, urgency accuracy `1.0000`, under-escalation rate `0.0000`
- `day1_fast_baseline`: mean reward `0.4723`, condition accuracy `0.1250`,
  urgency accuracy `0.1250`, under-escalation rate `0.8750`
- `oracle`: mean reward `0.9551`, condition accuracy `1.0000`, urgency
  accuracy `1.0000`, under-escalation rate `0.0000`

These numbers show that waiting for full evidence is much safer than one-shot
day-1 diagnosis on the checked-in trajectories. They do not yet prove that a
trained GRPO model beats the deterministic baseline.

Most relevant graphs:

- `final_1p5b_reward_chart.svg`
- `final_1p5b_quality_chart.svg`
- `final_1p5b_training_health_chart.svg`
- `baseline_vs_trained_benchmark_chart.svg`
- `evaluation_report.md`
- `demo_verification.md`

The baseline comparison graph is deliberately conservative: it compares the
checked-in legacy baseline average against the current 1.5B GRPO run's average,
best, and final benchmark scores, so judges can see both the real progress and
the remaining gap honestly.

Links:

- OpenEnv Space: `https://huggingface.co/spaces/sparsh122/maas-openenv`
- Patient + ASHA portal: `https://huggingface.co/spaces/nancyyyyyyy/niva-prenatal-health`
- Coordinator portal: `https://sparsh122-maternaai.hf.space/coordinator`
- Main README: `../README.md`
- Full blog: `../blog.md`
