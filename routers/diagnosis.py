import json
from html import escape
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from database import SessionLocal
from environment import ActionModel, PrenatalEnvironment
from models import DailyCheckin
from xai_reward_model import calculate_reward, choose_urgency, featurize, infer_reference_condition
from schemas import DiagnosisResponse

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])
env = PrenatalEnvironment()
DEFAULT_CHECKPOINT = Path("trained_models/maas_deep_policy.pt")


def _load_learned_prediction(user_id: int) -> dict[str, Any] | None:
    if not DEFAULT_CHECKPOINT.exists():
        return None

    try:
        # Keep Space startup lightweight unless a learned checkpoint is present.
        from maas_deep_policy import predict_for_user_id
    except Exception:
        return None

    try:
        return predict_for_user_id(
            user_id=user_id,
            checkpoint_path=DEFAULT_CHECKPOINT,
            db_path="prenatal.db",
        )
    except Exception:
        return None


def _render_diagnosis_html(
    *,
    user_id: int,
    predicted_condition: str,
    urgency: str,
    result,
    payload: DiagnosisResponse,
) -> HTMLResponse:
    pretty_json = escape(
        json.dumps(payload.model_dump(mode="json"), indent=2, default=str)
    )
    return HTMLResponse(
        f"""
        <html>
          <head>
            <title>MAAS Diagnosis {user_id}</title>
            <style>
              body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }}
              .card {{ max-width: 960px; background: white; border: 1px solid #cbd5e1; border-radius: 16px; padding: 24px; }}
              .chips {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 14px 0 18px; }}
              .chip {{ background: #e2e8f0; color: #1e293b; padding: 8px 12px; border-radius: 999px; font-weight: 600; }}
              pre {{ background: #0f172a; color: #e2e8f0; padding: 18px; border-radius: 12px; overflow: auto; }}
              p {{ color: #475569; line-height: 1.5; }}
            </style>
          </head>
          <body>
            <div class="card">
              <h1>MAAS Diagnosis Output</h1>
              <div class="chips">
                <span class="chip">User {user_id}</span>
                <span class="chip">Condition: {escape(predicted_condition)}</span>
                <span class="chip">Urgency: {escape(urgency)}</span>
                <span class="chip">Reward: {result.reward:.1f}</span>
              </div>
              <p>{escape(result.rationale)}</p>
              <pre>{pretty_json}</pre>
            </div>
          </body>
        </html>
        """
    )


@router.get("/{user_id}", response_model=DiagnosisResponse)
def diagnose_user(user_id: int, request: Request):
    try:
        prompt = env.reset(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    observation = prompt.observation
    features = featurize(observation)
    reference_condition = infer_reference_condition(observation)
    learned_prediction = _load_learned_prediction(user_id)

    predicted_condition = reference_condition
    urgency = choose_urgency(reference_condition, features)
    rationale = None
    if learned_prediction is not None:
        predicted_condition = learned_prediction["condition"]
        urgency = learned_prediction["urgency"]
        rationale = learned_prediction["rationale"]

    breakdown = calculate_reward(predicted_condition, urgency, observation)
    result = env.step(
        ActionModel(
            action_type="diagnose",
            condition=predicted_condition,
            urgency=urgency,
            rationale=rationale or breakdown.rationale,
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

    payload = DiagnosisResponse(
        user_id=user_id,
        predicted_condition=predicted_condition,
        urgency=urgency,
        rationale=result.rationale,
        reward=result.reward,
        risk_flags=observation.risk_flags,
        history_flags=observation.history_flags,
        diet_advice=result.diet_advice,
        days_of_data=observation.days_of_data,
        latest_checkin_at=latest_checkin_at,
    )
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return _render_diagnosis_html(
            user_id=user_id,
            predicted_condition=predicted_condition,
            urgency=urgency,
            result=result,
            payload=payload,
        )
    return payload


@router.get("/page/{user_id}", include_in_schema=False)
def diagnose_user_page(user_id: int, request: Request):
    payload = diagnose_user(user_id, request)
    if isinstance(payload, HTMLResponse):
        return payload
    return _render_diagnosis_html(
        user_id=user_id,
        predicted_condition=payload.predicted_condition,
        urgency=payload.urgency,
        result=type("DiagnosisView", (), {"reward": payload.reward, "rationale": payload.rationale})(),
        payload=payload,
    )
