from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import UserProfile
from schemas import UserProfileCreate, UserProfileResponse

router = APIRouter(prefix="/users", tags=["Users"])
@router.post("/setup", response_model=UserProfileResponse, status_code=201)
def create_user(user: UserProfileCreate, db: Session = Depends(get_db)):
    db_user = UserProfile(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/{user_id}", response_model=UserProfileResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User nahi mila")
    return user

