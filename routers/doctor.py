from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth_utils import ensure_seed_accounts, require_role
from database import get_db
from models import AuthAccount, UserProfile
from portal_services import build_patient_detail, ensure_escalation_task, get_or_create_review, list_patient_snapshots
from schemas import DoctorNoteUpdate

router = APIRouter(tags=["Doctor Portal"])
BASE_DIR = Path(__file__).resolve().parents[1]
DOCTOR_HTML = BASE_DIR / "doctor.html"


@router.get("/doctor", include_in_schema=False)
@router.get("/doctor.html", include_in_schema=False)
def doctor_portal():
    if not DOCTOR_HTML.exists():
        raise HTTPException(status_code=404, detail="doctor.html missing")
    return FileResponse(DOCTOR_HTML)


@router.get("/doctor/patients")
def list_doctor_patients(
    risk_level: str | None = Query(None),
    condition_flag: str | None = Query(None),
    last_checkin_date: date | None = Query(None),
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("doctor")),
):
    ensure_seed_accounts(db)
    patients = list_patient_snapshots(
        db,
        risk_level=risk_level,
        condition_flag=condition_flag,
        last_checkin_date=last_checkin_date,
    )
    banner_patients = [item for item in patients if item["escalated_recent"]]
    return {
        "banner_count": len(banner_patients),
        "banner_patients": banner_patients[:5],
        "patients": patients,
        "viewer": {"id": account.id, "name": account.display_name, "role": account.role},
    }


@router.get("/doctor/patients/{patient_id}")
def get_doctor_patient_detail(
    patient_id: int,
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("doctor")),
):
    profile = db.query(UserProfile).filter(UserProfile.id == patient_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Patient not found")
    ensure_seed_accounts(db)
    detail = build_patient_detail(db, patient_id)
    detail["viewer"] = {"id": account.id, "name": account.display_name}
    return detail


@router.patch("/doctor/patients/{patient_id}/notes")
def save_doctor_notes(
    patient_id: int,
    payload: DoctorNoteUpdate,
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("doctor")),
):
    review = get_or_create_review(db, patient_id)
    review.notes = payload.notes
    review.reviewed_by_id = account.id
    review.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(review)
    return {"saved": True, "updated_at": review.updated_at.isoformat() if review.updated_at else None}


@router.post("/doctor/patients/{patient_id}/review")
def mark_reviewed(
    patient_id: int,
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("doctor")),
):
    review = get_or_create_review(db, patient_id)
    review.reviewed = True
    review.reviewed_at = datetime.utcnow()
    review.reviewed_by_id = account.id
    review.updated_at = datetime.utcnow()
    db.commit()
    return build_patient_detail(db, patient_id)


@router.post("/doctor/patients/{patient_id}/escalate")
def escalate_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("doctor")),
):
    review = get_or_create_review(db, patient_id)
    review.urgency_override = "go_to_hospital_today"
    review.escalated_at = datetime.utcnow()
    review.reviewed = True
    review.reviewed_at = review.reviewed_at or datetime.utcnow()
    review.reviewed_by_id = account.id
    review.updated_at = datetime.utcnow()
    db.commit()
    ensure_escalation_task(db, patient_id)
    return build_patient_detail(db, patient_id)
