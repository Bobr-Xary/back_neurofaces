from typing import Optional
from pydantic import BaseModel, EmailStr
from app.models.enums import UserRole, DeviceType

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class RegisterUserRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: UserRole

class RegisterDeviceRequest(BaseModel):
    name: str
    type: DeviceType
    owner_user_id: Optional[str] = None  # UUID as string
    lat: Optional[float] = None
    lon: Optional[float] = None
    zone: Optional[str] = None

class DeviceSecretResponse(BaseModel):
    device_id: str
    device_secret: str  # show once
