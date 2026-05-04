# MAAS Strict Judge Audit

Generated during the submission hardening pass on 2026-05-04.

## Critical Blockers

- Live Space is reachable, but it is still serving the older reward schema. `results/demo_verification.md` shows all endpoints pass, while `Reward schema current` is `no`.
- No exact trained-model multi-turn before/after row is currently present in `results/evaluation_report.md`. The deterministic baseline and oracle are measured on the same eight trajectories, but trained GRPO improvement is not proven on that evaluator.
- `portal_services.py`, local databases, and artifact folders are local-only and must stay out of commits. `.gitignore` now covers them.

## High-Impact Weaknesses

- The deterministic visible baseline reaches the oracle on the checked-in eight trajectories, so the benchmark needs more adversarial cases before it can separate learned policies from robust heuristics.
- Training evidence is real but mixed: `results/final_1p5b_run_metrics.csv` has 18 logged steps and a best benchmark score of `0.255`, but final-step metrics do not show stable mastery.
- Public reward auditability depends on redeploying the updated environment. Until then, judges hitting the live Space will not see `adjusted_raw_reward` or trajectory penalty fields.

## Easy 2-Hour Fixes

- Redeploy `maas-openenv` from this repo and rerun `python scripts/verify_space.py`.
- Run `python scripts/check_submission_links.py` before submission and keep `results/submission_link_check.md`.
- Put `results/training_results_overview.svg` next to existing training plots in the README.
- Record a 90-second demo using `results/demo_video_script.md`.

## Medium-Depth Fixes

- Add 8-12 more multi-turn trajectories with adversarial splits: false reassurance, benign high-anxiety over-escalation, mixed BP/fetal movement, and rural access delay.
- Run `scripts/run_baseline_vs_trained.py` with actual trained action traces or a reachable trained model id.
- Export trained-model actions to JSONL so the exact same evaluator can compare baseline, trained, deterministic baseline, and oracle.

## Deep Research-Grade Improvements

- Add stochastic observation noise and access constraints that make "always wait until day 3" suboptimal for emergency cases.
- Train/evaluate an agent on held-out trajectories where condition labels are not directly implied by a single visible flag.
- Report calibration: escalation rate, under-escalation by urgency tier, over-escalation by low-risk cases, and evidence-gathering efficiency.

## Harsh Rubric Simulation

Minimum requirements:

| Requirement | Status | Evidence |
|---|---|---|
| OpenEnv dependency declared | MET | `pyproject.toml`, `requirements*.txt` use `openenv-core>=0.2.3` |
| HF TRL / Unsloth training path | MET | `train_grpo.py`, `train_grpo_multiturn.py`, notebooks |
| Real training loss + reward evidence | MET | `results/final_1p5b_run_metrics.csv`, charts, `results/grpo_training_summary.json` |
| Mini-blog or short video material | MET | `blog.md`, `results/mini_writeup.md`, `results/demo_video_script.md` |
| Environment hosted on HF Spaces | MET | `results/demo_verification.md` endpoint status pass |
| README links to materials | MOSTLY MET | README links canonical evidence; keep it tight |
| Exact trained multi-turn improvement | NOT MET | no trained row in `results/evaluation_report.md` |
| Live reward schema current | NOT MET | `results/demo_verification.md` says schema current `no` |

Estimated score after this pass:

- Environment Innovation: `33/40`
- Storytelling & Presentation: `25/30`
- Showing Improvement in Rewards: `11/20`
- Reward & Training Pipeline: `8/10`
- Total: `77/100`

Estimated rank: roughly top `80-150 / 800` if the judge values environment design, lower if exact trained improvement is weighted strictly.

Top 3 last-minute fixes:

1. Redeploy the Space so live reward components match local code.
2. Run trained model/action traces through `scripts/run_baseline_vs_trained.py`.
3. Add more held-out adversarial trajectories so the deterministic baseline no longer ties oracle.
