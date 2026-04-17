from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth_utils import ensure_seed_accounts, require_role
from database import get_db
from models import AuthAccount, CoordinatorTask, UserProfile
from portal_services import build_coordinator_dashboard, build_patient_detail
from schemas import CoordinatorTaskCreate, CoordinatorTaskUpdate

router = APIRouter(tags=["Coordinator Portal"])
BASE_DIR = Path(__file__).resolve().parents[1]
COORDINATOR_HTML = BASE_DIR / "coordinator.html"


@router.get("/coordinator", include_in_schema=False)
@router.get("/coordinator.html", include_in_schema=False)
def coordinator_portal():
    if not COORDINATOR_HTML.exists():
        raise HTTPException(status_code=404, detail="coordinator.html missing")
    return FileResponse(COORDINATOR_HTML)


@router.get("/coordinator/dashboard")
def coordinator_dashboard(
    status: str | None = Query(None),
    region: str | None = Query(None),
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("coordinator")),
):
    ensure_seed_accounts(db)
    payload = build_coordinator_dashboard(db, status=status, region=region)
    payload["viewer"] = {"id": account.id, "name": account.display_name, "role": account.role}
    return payload


@router.get("/coordinator/patients/{patient_id}")
def coordinator_patient_detail(
    patient_id: int,
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("coordinator")),
):
    profile = db.query(UserProfile).filter(UserProfile.id == patient_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Patient not found")
    detail = build_patient_detail(db, patient_id)
    detail["viewer"] = {"id": account.id, "name": account.display_name}
    return detail


@router.post("/coordinator/patients/{patient_id}/tasks")
def create_task(
    patient_id: int,
    payload: CoordinatorTaskCreate,
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("coordinator")),
):
    profile = db.query(UserProfile).filter(UserProfile.id == patient_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Patient not found")

    task = CoordinatorTask(
        patient_id=patient_id,
        owner_id=account.id,
        task_type=payload.task_type,
        title=payload.title,
        details=payload.details,
        priority=payload.priority,
        due_at=payload.due_at,
        status="open",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return {
        "created": True,
        "task": {
            "id": task.id,
            "title": task.title,
            "status": task.status,
            "priority": task.priority,
            "due_at": task.due_at.isoformat() if task.due_at else None,
        },
    }


@router.patch("/coordinator/tasks/{task_id}")
def update_task(
    task_id: int,
    payload: CoordinatorTaskUpdate,
    db: Session = Depends(get_db),
    account: AuthAccount = Depends(require_role("coordinator")),
):
    task = db.query(CoordinatorTask).filter(CoordinatorTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if payload.status is not None:
        task.status = payload.status
        if payload.status == "completed":
            task.completed_at = datetime.utcnow()
    if payload.details is not None:
        task.details = payload.details
    if payload.priority is not None:
        task.priority = payload.priority
    if payload.due_at is not None:
        task.due_at = payload.due_at

    task.owner_id = task.owner_id or account.id
    task.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(task)
    return {
        "updated": True,
        "task": {
            "id": task.id,
            "status": task.status,
            "priority": task.priority,
            "details": task.details or "",
            "due_at": task.due_at.isoformat() if task.due_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        },
    }
