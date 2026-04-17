from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Iterable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from database import get_db
from models import AuthAccount

JWT_SECRET = os.environ.get("PORTAL_JWT_SECRET", "niva-portal-local-secret")
TOKEN_TTL_SECONDS = int(os.environ.get("PORTAL_TOKEN_TTL_SECONDS", "43200"))

security = HTTPBearer(auto_error=False)

DEMO_ACCOUNTS = [
    {
        "email": "doctor@niva.local",
        "password": "doctor123",
        "role": "doctor",
        "display_name": "Dr. Meera Anand",
    },
    {
        "email": "coordinator@niva.local",
        "password": "care123",
        "role": "coordinator",
        "display_name": "Riya Singh",
    },
]


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    return hashlib.sha256(f"niva::{password}".encode("utf-8")).hexdigest()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("utf-8"))


def _encode_token(payload: dict) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_segment = _b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    signature = _b64encode(hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest())
    return f"{header_segment}.{payload_segment}.{signature}"


def _decode_token(token: str) -> dict:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format") from exc

    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    expected_signature = _b64encode(hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest())
    if not hmac.compare_digest(expected_signature, signature_segment):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")

    payload = json.loads(_b64decode(payload_segment))
    if payload.get("exp", 0) < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    return payload


def create_access_token(account: AuthAccount) -> str:
    payload = {
        "sub": str(account.id),
        "role": account.role,
        "email": account.email,
        "name": account.display_name,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    return _encode_token(payload)


def ensure_seed_accounts(db: Session) -> None:
    changed = False
    for item in DEMO_ACCOUNTS:
        account = (
            db.query(AuthAccount)
            .filter(AuthAccount.email == normalize_email(item["email"]))
            .first()
        )
        if account:
            continue
        db.add(
            AuthAccount(
                email=normalize_email(item["email"]),
                password_hash=hash_password(item["password"]),
                role=item["role"],
                display_name=item["display_name"],
                is_active=True,
            )
        )
        changed = True
    if changed:
        db.commit()


def get_current_account(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
) -> AuthAccount:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    payload = _decode_token(credentials.credentials)
    account = db.query(AuthAccount).filter(AuthAccount.id == int(payload["sub"])).first()
    if not account or not account.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account not available")
    return account


def require_role(*roles: str):
    allowed = set(roles)

    def dependency(account: AuthAccount = Depends(get_current_account)) -> AuthAccount:
        if account.role not in allowed:
            allowed_list = ", ".join(sorted(allowed))
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access restricted to roles: {allowed_list}",
            )
        return account

    return dependency


def role_matches(account: AuthAccount, roles: Iterable[str]) -> bool:
    return account.role in set(roles)
