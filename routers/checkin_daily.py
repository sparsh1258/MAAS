from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models import DailyCheckin, UserProfile
from schemas import DailyCheckinCreate, DailyCheckinResponse

router = APIRouter(prefix="/checkin", tags=["Daily Checkin"])

@router.post("/daily", response_model=DailyCheckinResponse, status_code=201)
def submit_daily_checkin(checkin: DailyCheckinCreate, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == checkin.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User nahi mila")
    db_checkin = DailyCheckin(**checkin.dict())
    db.add(db_checkin)
    db.commit()
    db.refresh(db_checkin)
    return db_checkin

@router.get("/daily/{user_id}", response_model=List[DailyCheckinResponse])
def get_daily_checkins(user_id: int, limit: int = 7, db: Session = Depends(get_db)):
    checkins = (
        db.query(DailyCheckin)
        .filter(DailyCheckin.user_id == user_id)
        .order_by(DailyCheckin.created_at.desc())
        .limit(limit)
        .all()
    )
    return checkins