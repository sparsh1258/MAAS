# Niva — AI Prenatal Health Monitor
Early risk detection for maternal health using AI + real-time monitoring

# Overview
**Niva** is an AI-powered prenatal health monitoring system designed to detect early warning signs in pregnancy and guide users toward safe, timely medical action. Niva transforms basic health signals into actionable risk insights.

# Problem
Maternal complications like:
- Preeclampsia : Affects 1 in 12 pregnancies globally
- Anemia :  Affects 14% of pregnancies worldwide
- Gestational Diabetes :   Affects 40% of pregnant women globally
- Preterm Labor

often go **undetected until it's too late**, especially where:
- medical access is limited  
- monitoring is inconsistent  
- awareness is low
  
# Solution
Niva acts as a **digital prenatal companion** that:
- Tracks daily health signals  
- Detects early risk patterns such as :

  - Preeclampsia : Critically high BP (≥160/110), headaches, swelling → emergency escalation 
  - Gestational Diabetes : Family history + low energy + breathlessness → PHC referral 
  - Anemia : Low nutrition markers + fatigue patterns 
  - Fetal Distress : Kick count < 3 in 2 hours → immediate hospital alert 
  - Preterm Risk : Early warning signals from symptom patterns
    
- Classifies condition severity  -
- Provides urgency-based guidance  
- Suggests diet + care recommendations  

# How It Works
1. User creates profile  
2. Daily / 3-day check-ins  
3. System collects signals:
   - BP, symptoms, hydration, sleep, kicks  
4. AI evaluates risk  
5. Outputs:
   - Condition class  
   - Urgency level
      - monitor_at_home : No immediate danger, continue daily check-ins 
      - visit_phc_this_week :  Concerning signs, needs professional check within days 
      - go_to_hospital_today : DANGER flags present, immediate medical attention required 

   - Recommendations  

# Key Features
- AI-based risk classification
- Real-time monitoring system
- Clean interactive frontend
- Daily + 3-day health tracking
- Urgency classification (SAFE / WATCH / DANGER)
- Smart diet recommendations
- OpenEnv RL-compatible environment
-  Evaluation system with graded tasks

# AI + OpenEnv Integration
Niva is not just an app — it's a **learning environment**.
It implements:
- `reset()`, `step()`, `state()` interaction model
- Structured observations + actions
- Reward-based evaluation
- LLM-based inference pipeline

# Tasks & Evaluation

| Task | Difficulty | Goal |
|------|-----------|------|
| Easy | Low | Basic pattern detection |
| Medium | Moderate | Multi-signal reasoning |
| Hard | High | Noisy/ambiguous decision making |

# Reward Logic
- Correct diagnosis → reward  
- Correct urgency → extra reward  
- Efficient decisions → bonus  
- Unsafe predictions → penalty
  
Final score range:0.0 → 1.0 

# Tech Stack
- **Backend**: FastAPI, SQLAlchemy, SQLite  
- **Frontend**: HTML, CSS, JS  
- **AI Layer**: OpenAI-compatible LLM  
- **Deployment**: Hugging Face Spaces (Docker)  
- **Environment**: OpenEnv RL-style  

# Project Structure
niva-prenatal-health/

├── inference.py          # Baseline agent — runs all tasks, emits judge-compliant logs

├── main.py               # FastAPI app entry 

├── openenv.yaml          # OpenEnv spec — tasks, action space, observation space

├── environment.py        # Environment logic

├── models.py             # SQLAlchemy database models

├── schemas.py            # Pydantic schemas

├── database.py           # DB connection setup

├── requirements.txt      # Python dependencies

├── Dockerfile            # Container config for HF Spaces

├── preview.html          # Frontend UI

├── tasks/

│   ├── __init__.py

│   ├── task_1_easy.py  # Preeclampsia danger task

│   ├── task_2_medium.py  # Fetal distress task

│   └── task_3_hard.py    # Gestational diabetes task

└── routers/

    ├── users.py
    
    ├── checkin_daily.py
    
    ├── checkin_3day.py
    
    └── diagnosis.py

# Live Demo
[Click here to try the app](https://huggingface.co/spaces/nancyyyyyyy/niva-prenatal-health)

# Environment Variables
| Variable | Description | Required |
|---|---|---|
| API_BASE_URL | LLM API endpoint (OpenAI-compatible) | Yes |
| MODEL_NAME | Model identifier (e.g. gpt-4o) | Yes |
| HF_TOKEN | Hugging Face / API key | Yes |

## Validation
OpenEnv pre-validation passed successfully.

## This project was developed by Muskaan Kohli, Nancy Garg and Sparsh Gupta as a team hackathon submission.
