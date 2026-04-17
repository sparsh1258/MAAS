from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth_utils import create_access_token, ensure_seed_accounts, get_current_account, hash_password, normalize_email
from database import get_db
from models import AuthAccount
from schemas import AuthAccountResponse, AuthLoginRequest, AuthLoginResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=AuthLoginResponse)
def login(payload: AuthLoginRequest, db: Session = Depends(get_db)):
    ensure_seed_accounts(db)
    email = normalize_email(payload.email)
    account = db.query(AuthAccount).filter(AuthAccount.email == email).first()
    if not account or not account.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if account.role != payload.role:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Incorrect role for this portal")
    if account.password_hash != hash_password(payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(account)
    return AuthLoginResponse(
        access_token=token,
        user=AuthAccountResponse.model_validate(account),
    )


@router.get("/me", response_model=AuthAccountResponse)
def me(account: AuthAccount = Depends(get_current_account)):
    return account
