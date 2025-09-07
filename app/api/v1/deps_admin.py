from fastapi import Depends, HTTPException, status, Header, Query
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional
# TODO: adjust import if needed
from app.core.config_env import settings
from app.db.session import SessionLocal  # должно существовать в проекте
from app.models.user import User  # модель пользователя
import re

BEARER_RE = re.compile(r"^Bearer\s+(?P<token>.+)$", re.IGNORECASE)

# Локальный зависимость get_db (если у вас уже есть - используйте её)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _extract_token(authorization: Optional[str], token_q: Optional[str]) -> str:
    if token_q:
        return token_q
    if authorization:
        m = BEARER_RE.match(authorization.strip())
        if m:
            return m.group("token")
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

def get_current_user(
    authorization: Optional[str] = Header(None, convert_underscores=False),
    token_q: Optional[str] = Query(None, alias="token"),
    db: Session = Depends(get_db),
) -> User:
    token = _extract_token(authorization, token_q)
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    # Предполагаем, что в токене хранится user_id в `sub` (при необходимости поменяйте claim)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No subject in token")
    try:
        user_uuid = UUID(str(user_id))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bad user id in token")
    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

def admin_required(user: User = Depends(get_current_user)) -> User:
    role = getattr(user, "role", None) or getattr(user, "role_name", None)
    if role not in {"admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user
