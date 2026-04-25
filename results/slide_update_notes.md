# Slide Update Notes

Use this as the short cleanup checklist for the final presentation.

## Slide 1: One-Line Thesis

- MAAS turns maternal triage into an OpenEnv professional-workflow task instead of a one-shot classifier.

## Slide 2: Why Theme 3.1

- Show the three core mechanics clearly: partial observability, multi-step actions, and temporal belief state across check-in days.
- Name the actions directly: `assess`, `request_signal`, `diagnose`.

## Slide 3: What Makes It Safe

- Highlight deterministic reward logic, class weighting, urgency alignment, and explicit under-escalation penalties.
- Make it obvious that dangerous under-escalation is the behavior the environment is designed to punish.

## Slide 4: What We Already Proved

- Use the checked-in PPO evidence: validation condition accuracy `0.9792`, urgency accuracy `1.0000`, baseline benchmark score `0.3367`.
- Mention that the training loop, verifier, and reward model were all exercised in the repo artifacts.

## Slide 5: Latest HF Training Status

- Reference job `69ed2261d70108f37acdef0e` and repo `sparsh122/maas-grpo-qwen05b-fix2`.
- Say the post-fix GRPO runs no longer suffer from the original `-20` reward collapse.
- Also say the mean benchmark score is still weak, so GRPO optimization remains in progress.

## Slide 6: Deployment Story

- Point to `Dockerfile`, `requirements-space.txt`, and the FastAPI endpoints `reset`, `step`, `state`, `health`.
- Describe the project as HF Jobs-ready now and Space-ready once the public app sync is finalized.

## Slide 7: Honest Close

- Do not claim that RL training is fully solved.
- Do claim that MAAS is a strong OpenEnv environment with real training hooks, real reward logic, and credible HF deployment infrastructure.
