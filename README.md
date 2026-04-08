# Niva — AI Prenatal Health Monitor
Early risk detection for maternal health using AI + real-time monitoring

## What is Niva?
Niva is an AI-powered prenatal health monitoring system that detects early warning signs during pregnancy and guides users toward timely medical action before complications escalate.
It combines daily health check-ins, symptom tracking, and an LLM-based inference pipeline to classify risk and generate urgency-based guidance in real time.

## The Problem
Maternal complications are common — and commonly missed:

| Condition | Prevalence |
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
| `monitor_at_home` | No immediate danger — continue daily check-ins |
| `visit_phc_this_week` | Concerning signs — see a doctor within days |
| `go_to_hospital_today` | DANGER — immediate medical attention required |

### Conditions Detected
- **Preeclampsia** — critically high BP (≥160/110), headaches, swelling → emergency escalation
- **Gestational Diabetes** — family history + low energy + breathlessness → PHC referral
- **Anemia** — low nutrition markers + fatigue patterns
- **Fetal Distress** — kick count < 3 in 2 hours → immediate hospital alert
- **Preterm Risk** — early warning from symptom combinations

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
├── inference.py           # Baseline agent — runs all tasks, emits judge-compliant logs
├── main.py                # FastAPI app entry
├── openenv.yaml           # OpenEnv spec — tasks, action space, observation space
├── environment.py         # RL-style environment logic
├── models.py              # SQLAlchemy database models
├── schemas.py             # Pydantic schemas
├── database.py            # DB connection setup
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container config for HF Spaces
├── preview.html           # Frontend UI
├── tasks/
│   ├── __init__.py
│   ├── task_1_easy.py     # Preeclampsia danger task
│   ├── task_2_medium.py   # Fetal distress task
│   └── task_3_hard.py     # Gestational diabetes task
└── routers/
    ├── users.py
    ├── checkin_daily.py
    ├── checkin_3day.py
    └── diagnosis.py
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
| Task 1 | Easy | Preeclampsia danger — basic pattern detection |
| Task 2 | Medium | Fetal distress — multi-signal reasoning |
| Task 3 | Hard | Gestational diabetes — noisy/ambiguous signals |

### Reward Logic
| Outcome | Effect |
| Correct condition diagnosis | + reward |
| Correct urgency level | + extra reward |
| Efficient decision-making | + bonus |
| Unsafe / incorrect predictions | − penalty |

**Final score range: `0.0 → 1.0`**

## Deployment (Hugging Face Spaces)
This project is containerised for HF Spaces using Docker.

### Environment Variables
| Variable | Description | Required |
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) | Yes |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o`) | Yes |
| `HF_TOKEN` | Hugging Face / API key | Yes |
Never commit API keys to the repo. Use Hugging Face Secrets for deployment.

## Tech Stack
| Layer | Technology |
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
MIT License — see [LICENSE](LICENSE) for details.
