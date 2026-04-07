from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from database import engine, Base
from models import UserProfile, DailyCheckin, Checkin3Day

Base.metadata.create_all(bind=engine)
BASE_DIR = Path(__file__).resolve().parent
PREVIEW_FILE = BASE_DIR / "preview.html"

app = FastAPI(
    title='Prenatal Health Monitor API',
    description='AI-powered prenatal health tracking system',
    version='1.0.0'
)

from routers import users, checkin_daily, checkin_3day, diagnosis

app.include_router(users.router)
app.include_router(checkin_daily.router)
app.include_router(checkin_3day.router)
app.include_router(diagnosis.router)

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
