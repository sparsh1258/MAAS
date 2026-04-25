---
title: Niva Prenatal Health
emoji: "🤰"
colorFrom: pink
colorTo: blue
sdk: docker
pinned: false
---

# Niva - AI Prenatal Health Monitor

Early risk detection for maternal health using AI and real-time monitoring.

## What is Niva?

Niva is an AI-powered prenatal health monitoring system that detects early warning signs during pregnancy and guides users toward timely medical action before complications escalate.

It combines daily health check-ins, symptom tracking, and an LLM-based inference pipeline to classify risk and generate urgency-based guidance in real time.

## The Problem

Maternal complications are common and commonly missed:

| Condition | Prevalence |
|--|--|
| Gestational Diabetes | ~40% of pregnancies |
| Anemia | ~14% of pregnancies |
| Preeclampsia | ~1 in 12 pregnancies |
| Preterm Labor | Significant cause of neonatal mortality |

These conditions often go undetected until they become emergencies, especially in areas with limited medical access, inconsistent monitoring, or low awareness.

## What Niva Does

Niva acts as a digital prenatal companion that:
- Collects daily health signals such as BP, symptoms, hydration, sleep, and fetal kick counts
- Detects early risk patterns using AI
- Classifies severity and assigns an urgency level
- Provides personalized diet and care recommendations

### Urgency Tiers

| Level | Meaning |
|--|--|
| `monitor_at_home` | No immediate danger - continue daily check-ins |
| `visit_phc_this_week` | Concerning signs - see a doctor within days |
| `go_to_hospital_today` | Danger - immediate medical attention required |

### Conditions Detected

- **Preeclampsia** - critically high BP (>=160/110), headaches, swelling -> emergency escalation
- **Gestational Diabetes** - family history + low energy + breathlessness -> PHC referral
- **Anemia** - low nutrition markers + fatigue patterns
- **Fetal Distress** - kick count < 3 in 2 hours -> immediate hospital alert
- **Preterm Risk** - early warning from symptom combinations

## Why This Is an RL Environment

This project is more than a health app. It is an OpenEnv-compatible maternal triage environment designed for verifiable RL.

It has the three core RL properties:
- The model acts step by step through structured outputs
- Success can be verified programmatically through deterministic reward logic
- The task is partially observable, because the agent only sees recent patient check-ins rather than full hidden clinical state

The environment exposes:
- `reset()` to start a new patient scenario
- `step()` to submit a diagnosis and urgency action
- `state()` to inspect the current environment state

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM API endpoint such as OpenAI, Together, or Groq

### 1. Clone the repo

```bash
git clone https://github.com/sparsh1258/MAAS.git
cd MAAS

### 2. Install dependencies
pip install -r requirements.txt

### 3. Set environment variables

export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="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"

4. Run the app

uvicorn main:app --reload
Open http://localhost:8000 to view the frontend.

API Endpoints
Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1}'

Step environment
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"diagnose","target":"preeclampsia","urgency":"go_to_hospital_today","rationale":"Critical BP with danger flags"}'
Get current state
curl http://localhost:8000/state
Running the AI Agent (Inference)
To run the baseline agent against all tasks and generate judge-compliant logs:

python inference.py
This runs all three evaluation tasks and outputs structured logs for scoring.

Training
A minimal TRL training scaffold is included in the repo.

Run PPO training with:

python train_trl.py --user-ids 1
An additional OpenEnv-style PPO loop is available in:

python train_openenv_ppo.py --user-ids 1
Project Structure
MAAS/
|-- inference.py           # Baseline agent - runs all tasks, emits judge-compliant logs
|-- main.py                # FastAPI app entry
|-- openenv.yaml           # OpenEnv spec - tasks, action space, observation space
|-- environment.py         # RL-style environment logic
|-- models.py              # SQLAlchemy database models
|-- schemas.py             # Pydantic schemas
|-- database.py            # DB connection setup
|-- requirements.txt       # Python dependencies
|-- Dockerfile             # Container config for HF Spaces
|-- preview.html           # Frontend UI
|-- tasks/
|   |-- __init__.py
|   |-- task_1_easy.py
|   |-- task_2_medium.py
|   |-- task_3_hard.py
|-- routers/
|   |-- users.py
|   |-- checkin_daily.py
|   |-- checkin_3day.py
|   |-- diagnosis.py
AI + OpenEnv Integration
Niva is not just an app - it is an RL-compatible learning environment.

The AI layer implements an OpenEnv-style interface:

env.reset()   # initialize a new patient scenario
env.step()    # submit an action (diagnosis + urgency)
env.state()   # observe current health signals
Each call to step() sends structured health observations to an LLM, which reasons over symptom patterns and returns a risk classification with an urgency level.

How Inference Works
Health signals such as BP, symptoms, and kick count are structured into a prompt
The LLM reasons over these signals against known risk patterns
It returns a condition label and urgency tier
The environment evaluates correctness and assigns a reward
Evaluation Tasks
Task	Difficulty	Scenario
Task 1	Easy	Preeclampsia danger - basic pattern detection
Task 2	Medium	Fetal distress - multi-signal reasoning
Task 3	Hard	Gestational diabetes - noisy or ambiguous signals
These tasks are designed as a simple curriculum:

Easy tasks provide a non-zero success starting point
Medium tasks require combining multiple signals
Hard tasks include ambiguity and competing risks
Reward Logic
Outcome	Effect
Correct condition diagnosis	Positive reward
Correct urgency level	Additional reward
More recent data	Small bonus
Unsafe or incorrect predictions	Penalty
The reward is computed programmatically using deterministic safety-aware logic in xai_reward_model.py.

Deployment (Hugging Face Spaces)
This project is containerized for Hugging Face Spaces using Docker.

Add your deployed Space link here:

HF Space: PASTE_YOUR_SPACE_URL_HERE
Add your blog or short video link here:

Project writeup/video: PASTE_YOUR_LINK_HERE
Environment Variables
Variable	Description	Required
API_BASE_URL	LLM API endpoint (OpenAI-compatible)	Yes
MODEL_NAME	Model identifier such as gpt-4o	Yes
HF_TOKEN	API key or Hugging Face token	Yes
Never commit API keys to the repo. Use Hugging Face Secrets for deployment.

Tech Stack
Layer	Technology
Backend	FastAPI, SQLAlchemy, SQLite
Frontend	HTML, CSS, JavaScript
AI Layer	OpenAI-compatible LLM
Deployment	Hugging Face Spaces (Docker)
Environment	OpenEnv-style RL environment
Training	Hugging Face TRL
Validation
OpenEnv pre-validation passed successfully.

Team
Built by Muskaan Kohli, Nancy Garg, and Sparsh Gupta as a hackathon submission.

License
MIT License - see LICENSE for details.
