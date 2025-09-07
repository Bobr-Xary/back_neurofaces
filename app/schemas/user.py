from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr

from app.models.enums import UserRole

# Pydantic v2: from_attributes; v1: orm_mode
try:
    from pydantic import ConfigDict
    _V2 = True
except Exception:
    ConfigDict = None
    _V2 = False


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool

    if _V2:
        # Pydantic v2
        model_config = ConfigDict(from_attributes=True)
    else:
        # Pydantic v1
        class Config:
            orm_mode = True
