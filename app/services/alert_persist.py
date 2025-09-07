# app/services/alert_persist.py
from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import uuid
import json

import cv2
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import SessionLocal, Base, engine
from app.models.alert import Alert

# Ленивая инициализация таблиц (на случай, если миграции ещё не прогнаны)
_tables_ready = False
def _ensure_tables():
    global _tables_ready
    if not _tables_ready:
        Base.metadata.create_all(bind=engine)
        _tables_ready = True

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _save_bytes(data: bytes, root: str) -> str:
    day = datetime.utcnow().strftime("%Y%m%d")
    out_dir = Path(root) / day
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.jpg"
    path = out_dir / name
    with open(path, "wb") as f:
        f.write(data)
    return str(path)

def _save_face_ndarray(face_bgr: np.ndarray, root: str) -> str:
    day = datetime.utcnow().strftime("%Y%m%d")
    out_dir = Path(root) / day
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.jpg"
    path = out_dir / name
    ok, buf = cv2.imencode(".jpg", face_bgr)
    if not ok:
        raise RuntimeError("Could not encode face image")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return str(path)

def create_alert_record(
    *,
    db: Optional[Session] = None,
    original_bytes: Optional[bytes],          # исходный кадр (полный), можно None
    face_bgr: Optional[np.ndarray],           # кроп лица (BGR), можно None
    device_id: Optional[str] = None,
    face_id: Optional[str] = None,
    emotion: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    address: Optional[str] = None,
    label: str = "aggression",
) -> Alert:
    """
    Создаёт запись Alert в БД. Возвращает ORM-объект.
    Поля:
      id, ts_utc, device_id, severity, label, face_id, emotion, meta, lat, lon, address, image_path, image_face_path
    """
    _ensure_tables()

    # Вычисляем severity из aggression_score
    severity = 0
    if isinstance(emotion, dict) and "aggression_score" in emotion:
        try:
            severity = int(round(float(emotion["aggression_score"])))
        except Exception:
            severity = 0
    severity = max(0, min(severity, 10))

    image_path = None
    if original_bytes:
        try:
            image_path = _save_bytes(original_bytes, "alerts/raw")
        except Exception:
            image_path = None

    image_face_path = None
    if face_bgr is not None:
        try:
            image_face_path = _save_face_ndarray(face_bgr, "alerts/faces")
        except Exception:
            image_face_path = None

    auto_session = db is None
    db = db or SessionLocal()
    try:
        alert = Alert(
            ts_utc=_now_utc(),
            device_id=device_id,
            severity=severity,
            label=label,
            face_id=face_id,
            emotion=emotion or {},
            meta=meta or {},
            lat=lat, lon=lon, address=address,
            image_path=image_path,
            image_face_path=image_face_path,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert
    except SQLAlchemyError:
        if db:
            db.rollback()
        raise
    finally:
        if auto_session:
            db.close()
