# app/api/v1/auth.py
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.models.enums import UserRole
from app.schemas.auth import LoginRequest, TokenPair, RefreshRequest, RegisterUserRequest
from app.core.security import verify_password, hash_password, hash_token  # ← оставляем только эти
from app.core.deps import require_role
from app.core.tokens import create_access_token, create_refresh_token     # ← генерация токенов только отсюда

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=TokenPair)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email, User.is_active == True).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    access = create_access_token(sub=str(user.id), role=user.role.value)
    refresh_raw, expires_at = create_refresh_token(sub=str(user.id))

    # сохраняем refresh как хеш + дата истечения (наивный UTC)
    rt = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(refresh_raw),
        expires_at=expires_at,
        revoked=False
    )
    db.add(rt)
    db.commit()

    return TokenPair(access_token=access, refresh_token=refresh_raw)

@router.post("/refresh", response_model=TokenPair)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    token_hash = hash_token(payload.refresh_token)
    now = datetime.utcnow()

    rt = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.revoked == False
    ).first()
    if not rt or rt.expires_at <= now:
        raise HTTPException(status_code=401, detail="Invalid/expired refresh token")

    user = db.query(User).filter(User.id == rt.user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # rotate refresh
    rt.revoked = True
    new_raw, expires_at = create_refresh_token(sub=str(user.id))
    new_rt = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(new_raw),
        expires_at=expires_at,
        revoked=False
    )
    db.add(new_rt)
    db.commit()

    access = create_access_token(sub=str(user.id), role=user.role.value)
    return TokenPair(access_token=access, refresh_token=new_raw)

@router.post("/register", dependencies=[Depends(require_role(UserRole.admin))])
def register_user(payload: RegisterUserRequest, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=payload.email,
        full_name=payload.full_name or "",
        password_hash=hash_password(payload.password),
        role=payload.role
    )
    db.add(user)
    db.commit()
    return {"id": str(user.id), "email": user.email, "role": user.role.value}
