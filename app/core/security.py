import time, secrets, hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple
from passlib.context import CryptContext
from jose import jwt
from app.core.config import JWT_SECRET, JWT_ALG, ACCESS_TOKEN_EXPIRES_MIN, REFRESH_TOKEN_EXPIRES_DAYS

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def create_access_token(sub: str, role: str, extra: Optional[dict] = None) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": sub,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_EXPIRES_MIN)).timestamp()),
        "type": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def create_refresh_token() -> Tuple[str, datetime]:
    now = datetime.utcnow()
    exp = now + timedelta(days=REFRESH_TOKEN_EXPIRES_DAYS)
    token = secrets.token_urlsafe(48)
    return token, exp
