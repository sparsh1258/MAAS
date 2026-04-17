from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from environment import ActionModel, Observation, PrenatalEnvironment, StepResult
from models import Checkin3Day, DailyCheckin, UserProfile
from schemas import ResetRequest

Base.metadata.create_all(bind=engine)
BASE_DIR = Path(__file__).resolve().parent
PREVIEW_FILE = BASE_DIR / "preview.html"
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
def healthcheck():
    return {"status": "ok"}


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


@app.post("/reset", response_model=Observation, tags=["OpenEnv"])
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
