
from pydantic import BaseModel
from typing import Optional

class ShareAlertRequest(BaseModel):
    user_id: str
    can_view_face: bool = False

class HideAlertRequest(BaseModel):
    hidden: bool = True
    reason: Optional[str] = None  # will be stored in meta
