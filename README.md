---
title: Niva Prenatal Health
emoji: "🤰"
colorFrom: pink
colorTo: blue
sdk: docker
pinned: false
---
# Niva â€” AI Prenatal Health Monitor
Early risk detection for maternal health using AI + real-time monitoring

## What is Niva?
Niva is an AI-powered prenatal health monitoring system that detects early warning signs during pregnancy and guides users toward timely medical action before complications escalate.
It combines daily health check-ins, symptom tracking, and an LLM-based inference pipeline to classify risk and generate urgency-based guidance in real time.

## The Problem
Maternal complications are common â€” and commonly missed:

| Condition | Prevalence |
|--|--|
| Gestational Diabetes | ~40% of pregnancies |
| Anemia | ~14% of pregnancies |
| Preeclampsia | ~1 in 12 pregnancies |
| Preterm Labor | Significant cause of neonatal mortality |

These conditions often go undetected until they become emergencies especially in areas with limited medical access, inconsistent monitoring, or low awareness.

## What Niva Does
Niva acts as a **digital prenatal companion** that:
- Collects daily health signals (BP, symptoms, hydration, sleep, fetal kick counts)
- Detects early risk patterns using AI
- Classifies severity and assigns an urgency level
- Provides personalised diet and care recommendations

### Urgency Tiers
| Level | Meaning |
|--|--|
| `monitor_at_home` | No immediate danger â€” continue daily check-ins |
| `visit_phc_this_week` | Concerning signs â€” see a doctor within days |
| `go_to_hospital_today` | DANGER â€” immediate medical attention required |

### Conditions Detected
- **Preeclampsia** â€” critically high BP (â‰¥160/110), headaches, swelling â†’ emergency escalation
- **Gestational Diabetes** â€” family history + low energy + breathlessness â†’ PHC referral
- **Anemia** â€” low nutrition markers + fatigue patterns
- **Fetal Distress** â€” kick count < 3 in 2 hours â†’ immediate hospital alert
- **Preterm Risk** â€” early warning from symptom combinations

## Getting Started
### Prerequisites
- Python 3.10+
- An OpenAI-compatible LLM API endpoint (e.g. OpenAI, Together, Groq)

### 1. Clone the repo
```bash
git clone https://github.com/your-username/niva-prenatal-health.git
cd niva-prenatal-health
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Set environment variables
```bash
export API_BASE_URL="https://api.openai.com/v1"   # or your LLM endpoint
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your_huggingface_token"
```
### 4. Run the app
```bash
uvicorn main:app --reload
```
Open `http://localhost:8000` to view the frontend.

## Running the AI Agent (Inference)
To run the baseline agent against all tasks and generate judge-compliant logs:

```bash
python inference.py
```
This runs all three evaluation tasks (easy, medium, hard) and outputs structured logs for scoring.

##  Project Structure

```
niva-prenatal-health/
â”œâ”€â”€ inference.py           # Baseline agent â€” runs all tasks, emits judge-compliant logs
â”œâ”€â”€ main.py                # FastAPI app entry
â”œâ”€â”€ openenv.yaml           # OpenEnv spec â€” tasks, action space, observation space
â”œâ”€â”€ environment.py         # RL-style environment logic
â”œâ”€â”€ models.py              # SQLAlchemy database models
â”œâ”€â”€ schemas.py             # Pydantic schemas
â”œâ”€â”€ database.py            # DB connection setup
â”œâ”€â”€ requirements.txt       # Python dependencies
â”œâ”€â”€ Dockerfile             # Container config for HF Spaces
â”œâ”€â”€ preview.html           # Frontend UI
â”œâ”€â”€ tasks/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ task_1_easy.py     # Preeclampsia danger task
â”‚   â”œâ”€â”€ task_2_medium.py   # Fetal distress task
â”‚   â””â”€â”€ task_3_hard.py     # Gestational diabetes task
â””â”€â”€ routers/
    â”œâ”€â”€ users.py
    â”œâ”€â”€ checkin_daily.py
    â”œâ”€â”€ checkin_3day.py
    â””â”€â”€ diagnosis.py
```

## AI + OpenEnv Integration
Niva is not just an app it's an **RL-compatible learning environment**.
The AI layer implements an OpenEnv-style interface:

```python
env.reset()   # initialise a new patient scenario
env.step()    # submit an action (diagnosis + urgency)
env.state()   # observe current health signals
```

Each call to `step()` sends structured health observations to an LLM (via the OpenAI-compatible API), which reasons over symptom patterns and returns a risk classification with urgency level.

### How the Inference Works
1. Health signals (BP, symptoms, kick count, etc.) are structured into a prompt
2. The LLM reasons over these signals against known risk patterns
3. Output: condition class + urgency tier + recommendations
4. The environment evaluates correctness and assigns a reward score

## Evaluation Tasks
| Task | Difficulty | Scenario |
|--|--|--|
| Task 1 | Easy | Preeclampsia danger â€” basic pattern detection |
| Task 2 | Medium | Fetal distress â€” multi-signal reasoning |
| Task 3 | Hard | Gestational diabetes â€” noisy/ambiguous signals |

### Reward Logic
| Outcome | Effect |
|--|--|
| Correct condition diagnosis | + reward |
| Correct urgency level | + extra reward |
| Efficient decision-making | + bonus |
| Unsafe / incorrect predictions | âˆ’ penalty |

**Final score range: `0.0 â†’ 1.0`**

## Deployment (Hugging Face Spaces)
This project is containerised for HF Spaces using Docker.

### Environment Variables
| Variable | Description | Required |
|--|--|--|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) | Yes |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o`) | Yes |
| `HF_TOKEN` | Hugging Face / API key | Yes |
Never commit API keys to the repo. Use Hugging Face Secrets for deployment.

## Tech Stack
| Layer | Technology |
|--|--|
| Backend | FastAPI, SQLAlchemy, SQLite |
| Frontend | HTML, CSS, JavaScript |
| AI Layer | OpenAI-compatible LLM (any provider) |
| Deployment | Hugging Face Spaces (Docker) |
| Environment | OpenEnv RL-style |

## Validation
OpenEnv pre-validation passed successfully.

## Team
Built by **Muskaan Kohli**, **Nancy Garg**, and **Sparsh Gupta** as a hackathon submission.

## License
MIT License â€” see [LICENSE](LICENSE) for details.


