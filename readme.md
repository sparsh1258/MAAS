# 🌸 Niva — AI Prenatal Health Monitor

Niva is an AI-powered prenatal health monitoring system designed to assist pregnant women—especially in low-resource settings—by tracking daily health data, identifying risk conditions early, and providing actionable guidance.

---

## 🚀 Overview

Niva combines:

* 📊 Health tracking (BP, nutrition, sleep, baby kicks)
* 🧠 Rule-based AI diagnosis engine (OpenEnv)
* 🎬 Gesture-driven modern UI (swipe, drag, physics-based interactions)

The system simulates real-world prenatal monitoring and provides **risk assessment + urgency recommendations** in real time.

---

## 🧩 Features

### 👩‍⚕️ Health Monitoring

* Daily check-ins (BP, symptoms, kicks, nutrition)
* 3-day trend tracking (weight, energy, breathlessness)
* Automatic trend detection (rising BP, low nutrition, etc.)

### 🧠 AI Risk Detection

* Detects:

  * Preeclampsia
  * Gestational Diabetes
  * Anemia
  * Preterm Risk
  * Fetal Distress
* Generates:

  * Risk flags
  * Condition classification
  * Urgency level (home / PHC / hospital)

### 🎯 Smart Alerts System

* Dynamic alerts based on health data
* Priority classification:

  * 🔴 Danger
  * 🟡 Warning
  * 🔵 Info
* Swipe-to-dismiss gesture system

### 🎬 Modern UI (Frontend)

* Built with React + Framer Motion
* Gesture-based interactions:

  * Swipe cards
  * Drag with physics
  * Smooth transitions
* iOS-style UI design (glassmorphism + depth)

---

## 🏗️ Tech Stack

### Frontend

* React (Vite)
* Framer Motion (animations)
* @use-gesture/react (gesture handling)

### Backend

* FastAPI
* SQLAlchemy
* Pydantic

### AI / Logic

* Rule-based engine (OpenEnv compatible)
* Risk classification + reward-based evaluation

### Database

* SQLite (`prenatal.db`)

---

## 📂 Project Structure

```
niva-app/
│
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── environment.py
│   ├── models.py
│   ├── schemas.py
│   ├── routers/
│   └── prenatal.db
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   └── styles.css
│   └── package.json
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/niva.git
cd niva
```

---

### 2️⃣ Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs on:

```
http://localhost:7860
```

---

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on:

```
http://localhost:5173
```

---

## 🔄 System Flow

1. User enters health data
2. Data stored in database
3. Environment builds observation
4. AI engine detects risk
5. Alerts generated
6. UI displays swipeable cards

---

## 🧠 AI Logic Highlights

* Uses **risk flags system**:

  * `DANGER_BP_CRITICAL`
  * `LOW_KICK_AVG`
  * `HIGH_PREECLAMPSIA_SIGNAL`
* Combines:

  * Current readings
  * Trends (3-day)
  * Medical history
* Outputs:

  * Condition
  * Urgency
  * Diet advice

---

## 🎯 Use Case

Designed for:

* Rural healthcare support
* ASHA worker assistance
* Early risk detection without lab tests
* Low-cost mobile health monitoring

---

## 🚀 Future Improvements

* Real ML model (instead of rules)
* Push notifications
* PWA (installable app)
* Real-time charts & analytics
* Cloud deployment (Hugging Face / Docker)

---

## 👨‍💻 Author

Sparsh Gupta
Thapar Institute of Engineering & Technology

---

## 📜 License

This project is for educational and research purposes.
