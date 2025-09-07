# app/core/deps.py
import os
import logging
from typing import Optional, Iterable
from uuid import UUID

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import cast, String
from jose import jwt, JWTError

from app.db.session import get_db
from app.models.user import User
from app.models.device import Device
from app.models.enums import UserRole
from app.core.config import JWT_SECRET, JWT_ALG

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

DEBUG_AUTH = os.getenv("DEBUG_AUTH", "0").lower() in ("1", "true", "yes")
logger = logging.getLogger("auth")
if DEBUG_AUTH:
    logging.basicConfig(level=logging.INFO)


def _log(msg: str, **kw):
    if DEBUG_AUTH:
        safe_kw = {k: (v if k != "token" else f"{str(v)[:16]}...") for k, v in kw.items()}
        logger.info("[auth] " + msg + " " + " ".join(f"{k}={v}" for k, v in safe_kw.items()))

def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    _log("incoming token", token=token, len=len(token) if token else 0)

    # 1) Декод токена
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALG],
            options={"verify_aud": False},  # не проверяем aud
        )
        _log("decoded claims", keys=list(payload.keys()))
        tok_type = payload.get("type")
        sub_raw = payload.get("sub")
        if tok_type != "access" or not sub_raw:
            _log("bad claims", type=tok_type, sub=sub_raw)
            raise credentials_exc
        sub_str = str(sub_raw)
        _log("claims ok", type=tok_type, sub=sub_str)
    except JWTError as e:
        _log("JWTError on decode", err=str(e))
        raise credentials_exc
    except Exception as e:
        _log("unknown error on decode", err=str(e))
        raise credentials_exc

    # 2) Поиск пользователя по UUID
    user = None
    try:
        user_uuid = UUID(sub_str)
        _log("try find by UUID", uuid=user_uuid)
        user = db.query(User).filter(User.id == user_uuid, User.is_active.is_(True)).first()
        _log("find by UUID result", found=bool(user))
    except Exception as e:
        _log("UUID parse failed", err=str(e))

    # 3) Поиск по строковому id
    if not user:
        _log("try find by string id", id=sub_str)
        user = (
            db.query(User)
            .filter(cast(User.id, String) == sub_str, User.is_active.is_(True))
            .first()
        )
        _log("find by string id result", found=bool(user))

    # 4) Поиск по email (на случай, если sub = email)
    if not user:
        _log("try find by email", email=sub_str)
        user = (
            db.query(User)
            .filter(User.email == sub_str, User.is_active.is_(True))
            .first()
        )
        _log("find by email result", found=bool(user))

    if not user:
        _log("user not found -> 401")
        raise credentials_exc

    _log("user resolved", user_id=user.id, email=user.email, role=getattr(user.role, "value", user.role))
    return user


def require_role(*allowed: Iterable[UserRole]):
    allowed_set = set(allowed or [])

    def dep(user: User = Depends(get_current_user)) -> User:
        if allowed_set and user.role not in allowed_set:
            _log("role denied", have=getattr(user.role, "value", user.role), need=[r.value for r in allowed_set])
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dep


def get_current_device(
    db: Session = Depends(get_db),
    x_device_id: Optional[str] = Header(None, alias="X-Device-Id"),
    x_device_secret: Optional[str] = Header(None, alias="X-Device-Secret"),
) -> Device:
    if not x_device_id or not x_device_secret:
        raise HTTPException(status_code=401, detail="Missing device credentials")

    device = None
    # UUID
    try:
        dev_uuid = UUID(str(x_device_id))
        _log("device try UUID", dev_uuid=dev_uuid)
        device = db.query(Device).filter(Device.id == dev_uuid, Device.is_active.is_(True)).first()
        _log("device by UUID", found=bool(device))
    except Exception as e:
        _log("device UUID parse fail", err=str(e))

    # строка
    if not device:
        _log("device try string id", id=x_device_id)
        device = (
            db.query(Device)
            .filter(cast(Device.id, String) == str(x_device_id), Device.is_active.is_(True))
            .first()
        )
        _log("device by string", found=bool(device))

    if not device:
        raise HTTPException(status_code=401, detail="Invalid device")

    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    if not pwd_context.verify(x_device_secret, device.secret_hash):
        raise HTTPException(status_code=401, detail="Invalid device secret")

    return device


def get_current_user_optional(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"verify_aud": False})
        if payload.get("type") != "access":
            return None
        sub_str = str(payload.get("sub") or "")
        if not sub_str:
            return None
        try:
            user_uuid = UUID(sub_str)
            user = db.query(User).filter(User.id == user_uuid, User.is_active.is_(True)).first()
        except Exception:
            user = db.query(User).filter(cast(User.id, String) == sub_str, User.is_active.is_(True)).first()
        if not user:
            user = db.query(User).filter(User.email == sub_str, User.is_active.is_(True)).first()
        return user
    except Exception:
        return None
