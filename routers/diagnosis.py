from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from database import SessionLocal
from environment import (
    ActionModel,
    CONDITION_URGENCY,
    DIET_ADVICE,
    PrenatalEnvironment,
    _classify_condition,
)
from models import DailyCheckin
from schemas import DiagnosisResponse

router = APIRouter(prefix='/diagnosis', tags=['Diagnosis'])
env = PrenatalEnvironment()

@router.get('/{user_id}', response_model=DiagnosisResponse)
def diagnose_user(user_id: int):
    try:
        observation = env.reset(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    predicted_condition, rationale = _classify_condition(observation)
    urgency = CONDITION_URGENCY[predicted_condition]
    result = env.step(
        ActionModel(
            action_type='diagnose',
            target=predicted_condition,
            urgency=urgency,
        )
    )

    latest_checkin_at = None
    db: Session = SessionLocal()
    try:
        latest_checkin = (
            db.query(DailyCheckin)
            .filter(DailyCheckin.user_id == user_id)
            .order_by(DailyCheckin.created_at.desc())
            .first()
        )
        latest_checkin_at = latest_checkin.created_at if latest_checkin else None
    finally:
        db.close()

    return DiagnosisResponse(
        user_id=user_id,
        predicted_condition=predicted_condition,
        urgency=urgency,
        rationale=rationale,
        reward=result.reward,
        risk_flags=observation.risk_flags,
        history_flags=observation.history_flags,
        diet_advice=result.diet_advice,
        days_of_data=observation.days_of_data,
        latest_checkin_at=latest_checkin_at,
    )
