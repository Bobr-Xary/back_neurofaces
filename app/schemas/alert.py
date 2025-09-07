from typing import Optional, Any, Dict
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel

# Совместимость pydantic v1/v2
try:
    from pydantic import ConfigDict
    _V2 = True
except Exception:  # pydantic v1
    ConfigDict = None
    _V2 = False


class AlertIngestRequest(BaseModel):
    # вход как и раньше, тут менять не обязательно
    ts: Optional[str] = None                # ISO-строка времени (опционально)
    location: Optional[dict] = None         # {'lat':..,'lon':..}
    image_b64: str
    meta: Optional[Dict[str, Any]] = None


class AlertOut(BaseModel):
    # ВАЖНО: типы соответствуют колонкам в БД/модели ORM
    id: UUID
    ts_utc: datetime
    device_id: Optional[UUID] = None
    severity: int
    label: str
    face_id: Optional[str] = None
    emotion: Optional[dict] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    address: Optional[str] = None
    image_path: Optional[str] = None
    image_face_path: Optional[str] = None

    if _V2:
        # pydantic v2
        model_config = ConfigDict(from_attributes=True)
    else:
        # pydantic v1
        class Config:
            orm_mode = True
