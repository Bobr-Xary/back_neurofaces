import secrets
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from passlib.context import CryptContext
from app.db.session import get_db
from app.models.device import Device
from app.models.enums import DeviceType, UserRole
from app.schemas.auth import RegisterDeviceRequest, DeviceSecretResponse
from app.core.deps import require_role

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
router = APIRouter(prefix="/devices", tags=["devices"])

@router.post("/register", response_model=DeviceSecretResponse, dependencies=[Depends(require_role(UserRole.admin))])
def register_device(payload: RegisterDeviceRequest, db: Session = Depends(get_db)):
    secret_plain = secrets.token_urlsafe(32)
    secret_hash = pwd_context.hash(secret_plain)
    device = Device(
        name=payload.name,
        type=payload.type,
        secret_hash=secret_hash,
        owner_user_id=payload.owner_user_id,
        lat=payload.lat, lon=payload.lon, zone=payload.zone,
    )
    db.add(device); db.commit(); db.refresh(device)
    return DeviceSecretResponse(device_id=str(device.id), device_secret=secret_plain)
