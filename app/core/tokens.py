# app/core/tokens.py
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Tuple
from jose import jwt
import os

from app.core.config import JWT_SECRET, JWT_ALG

# Можно переопределить через env
ACCESS_TTL_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

def _now_utc_naive() -> datetime:
    # используем наивный UTC, чтобы сопоставлять с datetime.utcnow()
    return datetime.utcnow()

def _to_epoch_seconds(dt_naive_utc: datetime) -> int:
    return int(dt_naive_utc.replace(tzinfo=timezone.utc).timestamp())

def create_access_token(sub: str, role: str, extra: Dict[str, Any] | None = None) -> str:
    """
    Возвращает JWT-строку. iat/exp — числовые секунды, UTC.
    """
    now = _now_utc_naive()
    exp = now + timedelta(minutes=ACCESS_TTL_MIN)
    payload: Dict[str, Any] = {
        "sub": sub,
        "role": role,
        "type": "access",
        "iat": _to_epoch_seconds(now),
        "exp": _to_epoch_seconds(exp),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def create_refresh_token(sub: str) -> Tuple[str, datetime]:
    """
    Возвращает пару (refresh_token, expires_at_utc_naive).
    """
    now = _now_utc_naive()
    exp = now + timedelta(days=REFRESH_TTL_DAYS)
    payload = {
        "sub": sub,
        "type": "refresh",
        "iat": _to_epoch_seconds(now),
        "exp": _to_epoch_seconds(exp),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return token, exp
