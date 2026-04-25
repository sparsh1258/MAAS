---
title: Niva Prenatal Health
colorFrom: pink
colorTo: blue
sdk: docker
pinned: false
---

# Niva - AI Prenatal Health Monitor

Early risk detection for maternal health using AI and real-time monitoring.

---

## What is Niva?

Niva is an AI-powered prenatal health monitoring system that detects early warning signs during pregnancy and guides users toward timely medical action before complications escalate.

It combines daily health check-ins, symptom tracking, and an LLM-based inference pipeline to classify risk and generate urgency-based guidance in real time.

---

## The Problem

Maternal complications are common and often missed:

| Condition | Prevalence |
|----------|------------|
| Gestational Diabetes | ~40% of pregnancies |
| Anemia | ~14% of pregnancies |
| Preeclampsia | ~1 in 12 pregnancies |
| Preterm Labor | Major cause of neonatal mortality |

These conditions frequently go undetected until they become emergencies, especially in regions with limited healthcare access, inconsistent monitoring, or low awareness.

---

## What Niva Does

Niva acts as a digital prenatal companion that:

- Collects daily health signals such as blood pressure, symptoms, hydration, sleep, and fetal kick counts  
- Detects early risk patterns using AI  
- Classifies severity and assigns an urgency level  
- Provides personalized care and diet recommendations  

---

## Urgency Tiers

| Level | Meaning |
|------|--------|
| `monitor_at_home` | No immediate danger, continue monitoring |
| `visit_phc_this_week` | Concerning signs, consult a doctor soon |
| `go_to_hospital_today` | High risk, immediate medical attention required |

---

## Conditions Detected

- Preeclampsia: high blood pressure (>=160/110), headaches, swelling  
- Gestational Diabetes: fatigue, breathlessness, family history  
- Anemia: low nutrition indicators and fatigue patterns  
- Fetal Distress: kick count < 3 in 2 hours  
- Preterm Risk: early detection from combined symptom patterns  

---

## Why This Is an RL Environment

This project is an OpenEnv-compatible reinforcement learning environment designed for verifiable decision-making.

### Core Properties

- Step-by-step structured decision-making  
- Deterministic reward evaluation  
- Partially observable system  

### Environment Interface

```python
env.reset()   # initialize a new patient scenario
env.step()    # submit diagnosis and urgency action
env.state()   # observe current health signals
```

---

## Getting Started

### Prerequisites

- Python 3.10 or above  
- OpenAI-compatible API (OpenAI, Together, or Groq)

---

### 1. Clone the Repository

```bash
git clone https://github.com/sparsh1258/MAAS.git
cd MAAS
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Set Environment Variables

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your_token_here"
```

Do not commit API keys to the repository.

You can also use the example config file already included in the repo:

`.env.example`

---

### 4. Run the Application

```bash
uvicorn main:app --reload
```

Open in browser:

http://localhost:8000

---

## API Endpoints

### Reset Environment

```bash
curl -X POST http://localhost:8000/reset \
-H "Content-Type: application/json" \
-d '{"user_id": 1}'
```

---

### Step Environment

```bash
curl -X POST http://localhost:8000/step \
-H "Content-Type: application/json" \
-d '{"action_type":"diagnose","target":"preeclampsia","urgency":"go_to_hospital_today","rationale":"Critical BP with danger flags"}'
```

---

### Get Current State

```bash
curl http://localhost:8000/state
```

---

## Running the AI Agent

```bash
python inference.py
```

---

## Training

### TRL PPO Training

```bash
python train_trl.py --user-ids 1
```

### OpenEnv PPO Training

```bash
python train_openenv_ppo.py --user-ids 1
```

---

## Project Structure

```text
MAAS/
│── inference.py
│── main.py
│── openenv.yaml
│── environment.py
│── models.py
│── schemas.py
│── database.py
│── requirements.txt
│── Dockerfile
│── preview.html
│
├── tasks/
│   ├── task_1_easy.py
│   ├── task_2_medium.py
│   ├── task_3_hard.py
│
├── routers/
│   ├── users.py
│   ├── checkin_daily.py
│   ├── checkin_3day.py
│   ├── diagnosis.py
```

---

## How Inference Works

1. Health signals are structured into input  
2. The LLM analyzes symptom patterns  
3. Outputs condition and urgency level  
4. The environment evaluates correctness and assigns reward  

---

## Evaluation Tasks

| Task | Difficulty | Scenario |
|------|-----------|----------|
| Task 1 | Easy | Preeclampsia detection |
| Task 2 | Medium | Fetal distress |
| Task 3 | Hard | Gestational diabetes with noisy signals |

---

## Reward Logic

| Outcome | Effect |
|--------|--------|
| Correct diagnosis | Positive reward |
| Correct urgency | Additional reward |
| Recent data | Bonus |
| Incorrect or unsafe output | Penalty |

---

## Deployment

https://huggingface.co/spaces/nancyyyyyyy/niva-prenatal-health

---

## Tech Stack

| Layer | Technology |
|------|-----------|
| Backend | FastAPI, SQLAlchemy |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| AI Layer | OpenAI-compatible LLM |
| Deployment | Hugging Face Spaces (Docker) |
| RL Framework | OpenEnv + TRL |

---

## Team

- Muskaan Kohli  
- Nancy Garg  
- Sparsh Gupta  

---

## License

MIT License
