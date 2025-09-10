
from pydantic import BaseModel, Field
from typing import Optional

class SystemSettingsOut(BaseModel):
    telegram_enabled: bool = Field(True)
    telegram_chat_id: Optional[str] = None

class SystemSettingsUpdate(BaseModel):
    telegram_enabled: Optional[bool] = None
    telegram_chat_id: Optional[str] = None
