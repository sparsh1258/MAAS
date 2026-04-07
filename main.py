from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from models import UserProfile, DailyCheckin, Checkin3Day

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Prenatal Health Monitor API",
    description="AI-powered prenatal health tracking system",
    version="1.0.0"
)

from routers import users, checkin_daily, checkin_3day

app.include_router(users.router)
app.include_router(checkin_daily.router)
app.include_router(checkin_3day.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #http://localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Prenatal Health Monitor API is running"}