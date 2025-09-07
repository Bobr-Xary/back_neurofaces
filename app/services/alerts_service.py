from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.models.alert import Alert

def create_alert(
    db: Session,
    *,
    device_id: Optional[str],
    severity: int,
    label: str,
    face_id: Optional[str],
    emotion: Optional[Dict[str, Any]],
    lat: Optional[float],
    lon: Optional[float],
    address: Optional[str],
    image_path: Optional[str],
    image_face_path: Optional[str],
    meta: Optional[Dict[str, Any]] = None,
) -> Alert:
    alert = Alert(
        device_id=device_id,
        severity=severity,
        label=label,
        face_id=face_id,
        emotion=emotion,
        lat=lat,
        lon=lon,
        address=address,
        image_path=image_path,
        image_face_path=image_face_path,
        meta=meta or {},
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert
