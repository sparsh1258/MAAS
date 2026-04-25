from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from database import SessionLocal
from environment import ActionModel, PrenatalEnvironment
from models import DailyCheckin
from xai_reward_model import calculate_reward, choose_urgency, featurize, infer_reference_condition
from schemas import DiagnosisResponse

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])
env = PrenatalEnvironment()


@router.get("/{user_id}", response_model=DiagnosisResponse)
def diagnose_user(user_id: int):
    try:
        prompt = env.reset(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    observation = prompt.observation
    features = featurize(observation)
    reference_condition = infer_reference_condition(observation)
    urgency = choose_urgency(reference_condition, features)
    breakdown = calculate_reward(reference_condition, urgency, observation)

    result = env.step(
        ActionModel(
            action_type="diagnose",
            target=reference_condition,
            urgency=urgency,
            rationale=breakdown.rationale,
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
        predicted_condition=reference_condition,
        urgency=urgency,
        rationale=result.rationale,
        reward=result.reward,
        risk_flags=observation.risk_flags,
        history_flags=observation.history_flags,
        diet_advice=result.diet_advice,
        days_of_data=observation.days_of_data,
        latest_checkin_at=latest_checkin_at,
    )
