from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models import Checkin3Day, UserProfile
from schemas import Checkin3DayCreate, Checkin3DayResponse

router = APIRouter(prefix="/checkin", tags=["3Day Checkin"])

@router.post("/3day", response_model=Checkin3DayResponse, status_code=201)
def submit_3day_checkin(checkin: Checkin3DayCreate, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == checkin.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User nahi mila")
    db_checkin = Checkin3Day(**checkin.dict())
    db.add(db_checkin)
    db.commit()
    db.refresh(db_checkin)
    return db_checkin

@router.post("/3day", response_model=Checkin3DayResponse, status_code=201)
def submit_3day_checkin(checkin: Checkin3DayCreate, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == checkin.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User nahi mila")
    db_checkin = Checkin3Day(**checkin.dict())
    db.add(db_checkin)
    db.commit()
    db.refresh(db_checkin)
    return db_checkin

@router.get("/3day/{user_id}", response_model=List[Checkin3DayResponse])
def get_3day_checkins(user_id: int, limit: int = 5, db: Session = Depends(get_db)):
    checkins = (
        db.query(Checkin3Day)
        .filter(Checkin3Day.user_id == user_id)
        .order_by(Checkin3Day.created_at.desc())
        .limit(limit)
        .all()
    )
    return checkins

from routers import users, checkin_daily, checkin_3day

