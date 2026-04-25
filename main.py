from pathlib import Path

import json
from html import escape

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from environment import ActionModel, PrenatalEnvironment, PromptObservation, StepResult
from models import Checkin3Day, DailyCheckin, UserProfile
from schemas import ResetRequest

Base.metadata.create_all(bind=engine)
BASE_DIR = Path(__file__).resolve().parent
PREVIEW_FILE = BASE_DIR / "preview.html"
MAP_DATA_FILE = BASE_DIR / "india.json"
TRAINING_GRAPH_FILE = BASE_DIR / "results" / "maas_deep_policy_demo" / "training_curve.png"
TRAINING_SUMMARY_FILE = BASE_DIR / "results" / "maas_deep_policy_demo" / "demo_summary.json"
openenv_env = PrenatalEnvironment()

app = FastAPI(
    title='Prenatal Health Monitor API',
    description='AI-powered prenatal health tracking system',
    version='1.0.0'
)

from routers import auth, checkin_3day, checkin_daily, coordinator, diagnosis, doctor, users

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(checkin_daily.router)
app.include_router(checkin_3day.router)
app.include_router(diagnosis.router)
app.include_router(doctor.router)
app.include_router(coordinator.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/", include_in_schema=False)
def root():
    if PREVIEW_FILE.exists():
        return FileResponse(PREVIEW_FILE)
    return {"message": "Prenatal Health Monitor API is running"}

@app.get("/health", tags=["System"])
def healthcheck(request: Request):
    payload = {"status": "healthy"}
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(
            """
            <html>
              <head>
                <title>MAAS Health</title>
                <style>
                  body { font-family: Segoe UI, Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }
                  .card { max-width: 520px; background: white; border: 1px solid #cbd5e1; border-radius: 16px; padding: 24px; }
                  .pill { display: inline-block; background: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 999px; font-weight: 600; }
                  p { color: #475569; }
                </style>
              </head>
              <body>
                <div class="card">
                  <h1>MAAS Health Check</h1>
                  <p class="pill">Status: healthy</p>
                  <p>The MAAS server is running and responding.</p>
                </div>
              </body>
            </html>
            """
        )
    return payload


@app.get("/health-page", include_in_schema=False)
def health_page():
    return HTMLResponse(
        """
        <html>
          <head>
            <title>MAAS Health</title>
            <style>
              body { font-family: Segoe UI, Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }
              .card { max-width: 520px; background: white; border: 1px solid #cbd5e1; border-radius: 16px; padding: 24px; }
              .pill { display: inline-block; background: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 999px; font-weight: 600; }
              p { color: #475569; }
            </style>
          </head>
          <body>
            <div class="card">
              <h1>MAAS Health Check</h1>
              <p class="pill">Status: healthy</p>
              <p>The MAAS server is running and responding.</p>
            </div>
          </body>
        </html>
        """
    )


@app.get("/india-map", include_in_schema=False)
def india_map():
    if not MAP_DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="India map data not found")
    return FileResponse(MAP_DATA_FILE, media_type="application/geo+json")


@app.get("/training-graph", include_in_schema=False)
def training_graph():
    if not TRAINING_GRAPH_FILE.exists():
        raise HTTPException(status_code=404, detail="Training graph not found")
    return FileResponse(TRAINING_GRAPH_FILE, media_type="image/png")


@app.get("/training-summary", include_in_schema=False)
def training_summary():
    if not TRAINING_SUMMARY_FILE.exists():
        raise HTTPException(status_code=404, detail="Training summary not found")
    return FileResponse(TRAINING_SUMMARY_FILE, media_type="application/json")


@app.get("/training-report", include_in_schema=False)
def training_report():
    if not TRAINING_SUMMARY_FILE.exists():
        raise HTTPException(status_code=404, detail="Training summary not found")
    summary = json.loads(TRAINING_SUMMARY_FILE.read_text(encoding="utf-8"))
    final_epoch = summary.get("final_epoch", {})
    metrics_json = escape(json.dumps(final_epoch, indent=2))
    return HTMLResponse(
        f"""
        <html>
          <head>
            <title>MAAS Training Report</title>
            <style>
              body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }}
              .card {{ max-width: 1080px; background: white; border: 1px solid #cbd5e1; border-radius: 16px; padding: 24px; }}
              .chips {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0 20px; }}
              .chip {{ background: #e2e8f0; color: #1e293b; padding: 8px 12px; border-radius: 999px; font-weight: 600; }}
              img {{ width: 100%; max-width: 980px; border: 1px solid #cbd5e1; border-radius: 12px; }}
              pre {{ background: #0f172a; color: #e2e8f0; padding: 18px; border-radius: 12px; overflow: auto; }}
            </style>
          </head>
          <body>
            <div class="card">
              <h1>MAAS Training Report</h1>
              <div class="chips">
                <span class="chip">Val condition acc: {final_epoch.get("val_condition_acc", "n/a")}</span>
                <span class="chip">Val urgency acc: {final_epoch.get("val_urgency_acc", "n/a")}</span>
                <span class="chip">Val loss: {final_epoch.get("val_loss", "n/a")}</span>
              </div>
              <p>Training curve from the current MAAS deep-policy demo run.</p>
              <img src="/training-graph" alt="MAAS training curve" />
              <h2>Final Epoch Metrics</h2>
              <pre>{metrics_json}</pre>
            </div>
          </body>
        </html>
        """
    )


def _ensure_demo_user(db: Session) -> int:
    user = db.query(UserProfile).order_by(UserProfile.id.asc()).first()
    if user:
        return user.id

    demo_user = UserProfile(
        name="Demo User",
        age=26,
        height_cm=158,
        weight_kg=55,
        region="Punjab",
        weeks_pregnant=30,
        history_diabetes=False,
        history_hypertension=False,
        history_preeclampsia=False,
        history_prev_comp=False,
    )
    db.add(demo_user)
    db.commit()
    db.refresh(demo_user)

    daily = DailyCheckin(
        user_id=demo_user.id,
        weeks_pregnant_at_checkin=30,
        bp_systolic=118,
        bp_diastolic=76,
        kick_count=8,
        kick_count_normal=True,
        symptom_headache=False,
        symptom_blurred_vision=False,
        symptom_swelling=False,
        symptom_abdominal_pain=False,
        symptom_bleeding=False,
        symptom_dizziness=False,
        meals_count=3,
        water_litres=2.0,
        sleep_hours=7.0,
        notes="Auto-generated demo data",
    )
    trend = Checkin3Day(
        user_id=demo_user.id,
        weeks_pregnant_at_checkin=30,
        weight_kg=55,
        energy_level=6,
        breathlessness=2,
        notes="Auto-generated demo trend",
    )
    db.add(daily)
    db.add(trend)
    db.commit()
    return demo_user.id


def _resolve_user_id(request: ResetRequest) -> int:
    if request.user_id is not None:
        return request.user_id

    db = SessionLocal()
    try:
        return _ensure_demo_user(db)
    finally:
        db.close()


@app.post("/reset", response_model=PromptObservation, tags=["OpenEnv"])
def reset_environment(request: ResetRequest | None = None):
    payload = request or ResetRequest()
    user_id = _resolve_user_id(payload)
    try:
        return openenv_env.reset(user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult, tags=["OpenEnv"])
def step_environment(action: ActionModel):
    try:
        return openenv_env.step(action)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", tags=["OpenEnv"])
def environment_state():
    return openenv_env.state()
