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

The strongest current evidence shows that the online RL loop ran end to end with
non-zero gradients and non-zero reward variance on real runs. The honest caveat
is that the latest checked-in evidence proves trainability more strongly than it
proves stable final policy superiority on the newest multi-turn benchmark.

Most relevant graphs:

- `final_1p5b_reward_chart.svg`
- `final_1p5b_quality_chart.svg`
- `final_1p5b_training_health_chart.svg`

Links:

- OpenEnv Space: `https://huggingface.co/spaces/sparsh122/maas-openenv`
- Patient + ASHA portal: `https://huggingface.co/spaces/nancyyyyyyy/niva-prenatal-health`
- Coordinator portal: `https://sparsh122-maternaai.hf.space/coordinator`
- Main README: `../README.md`
- Full blog: `../blog.md`
