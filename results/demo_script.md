# 30-60 Second Demo Script

## Short Spoken Version

"This is MAAS, a multi-step maternal triage benchmark for OpenEnv Theme 3.1. Instead of a one-shot classifier, the agent works through a three-day prenatal trajectory with hidden information. It can request a BP recheck, request kick count, advance the episode, refer to PHC, or diagnose. The reward is safety-first, so under-escalating danger cases is penalized more heavily than asking for more evidence. In this repo we provide the OpenEnv manifest, FastAPI runtime, judge-compatible inference script, and checked-in reward and loss curves showing real training runs. The main contribution is a deployable multi-turn maternal-health environment that evaluates workflow behavior under uncertainty."

## 45-Second Timed Version

### 0-10s

"MAAS is an OpenEnv maternal-health benchmark for multi-turn prenatal triage."

### 10-20s

"The agent does not see everything at once. It reasons over a three-day trajectory with hidden signals and a carried belief state."

### 20-32s

"Instead of only diagnosing, it can request blood-pressure confirmation, request kick count, advance to the next day, refer to PHC, or finish with a final diagnosis."

### 32-45s

"The reward is safety-first: correct urgency matters, and missing danger cases is costly. We include the FastAPI environment, `openenv.yaml`, judge-style inference, and checked-in training evidence with reward and loss curves."

### 45-60s

"So the core contribution is not just a health model. It is a deployable OpenEnv benchmark for safe workflow reasoning under uncertainty."
