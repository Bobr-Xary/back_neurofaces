# app/services/alert_persist.py
from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import uuid

import cv2
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from psycopg2.extras import Json  # корректная передача dict в JSONB

from app.db.session import SessionLocal, Base, engine
from app.models.alert import Alert  # регистрируем модель в metadata


# ---------- ВСПОМОГАТЕЛЬНЫЕ ----------

_tables_ready = False
def _ensure_tables() -> None:
    """Ленивая инициализация таблиц (полезно в dev, если миграции ещё не прогнаны)."""
    global _tables_ready
    if not _tables_ready:
        Base.metadata.create_all(bind=engine)
        _tables_ready = True

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _to_jsonable(obj: Any) -> Any:
    """Рекурсивно приводит numpy-типы к обычным python-типаM, чтобы json не падал."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def _as_uuid(val: Optional[Any]) -> Optional[uuid.UUID]:
    """Пытается привести строку к UUID; иначе None."""
    if val is None:
        return None
    if isinstance(val, uuid.UUID):
        return val
    if isinstance(val, str):
        try:
            return uuid.UUID(val)
        except ValueError:
            return None
    return None

def _as_float(val: Optional[Any]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


# Базовые директории хранения файлов (в папке проекта)
RAW_DIR = Path("alerts/raw")
FACE_DIR = Path("alerts/faces")
RAW_DIR.mkdir(parents=True, exist_ok=True)
FACE_DIR.mkdir(parents=True, exist_ok=True)

def _save_jpg_bytes(dst_path: Path, data: bytes) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(data)

def _save_bgr(dst_path: Path, bgr: np.ndarray) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# ---------- ОСНОВНАЯ ФУНКЦИЯ ----------

def create_alert_record(
    *,
    original_bytes: bytes,
    face_bgr: np.ndarray,
    emotion: Dict[str, Any],
    label: str,
    device_id: Optional[str] = None,
    user_id: Optional[str] = None,
    face_id: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    address: Optional[str] = None,
    zone: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    captured_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Сохраняет файлы (в папке проекта) и создаёт запись в БД.
    Вставляет ТОЛЬКО те колонки, которые реально существуют в таблице alerts.
    Возвращает словарь с id, face_id и корректными URL для фронта.
    """
    _ensure_tables()

    # ---- 1) Файлы (структура папок: alerts/raw/{YYYYMMDD}/..., alerts/faces/{YYYYMMDD}/...)
    now = captured_at or _now_utc()
    day = now.astimezone(timezone.utc).strftime("%Y%m%d")
    rand = uuid.uuid4().hex[:8]
    raw_name = f"{day}_{rand}.jpg"
    face_name = f"{day}_{rand}_face.jpg"

    raw_abs = RAW_DIR / day / raw_name
    face_abs = FACE_DIR / day / face_name

    _save_jpg_bytes(raw_abs, original_bytes)
    _save_bgr(face_abs, face_bgr)

    # относительные пути, которые (по желанию) можно хранить в БД
    # ВАЖНО: без префикса "alerts/" — только поддеревья.
    raw_rel = f"raw/{day}/{raw_name}"
    face_rel = f"faces/{day}/{face_name}"

    # Полные URL для фронта (фронт смотрит именно на них)
    raw_url = f"/media/alerts/{raw_rel}"
    face_url = f"/media/alerts/{face_rel}"

    # ---- 2) Подготовка данных
    # severity (если колонка есть — вставим)
    try:
        sev = int(round(float(emotion.get("aggression_score", 0))))
    except Exception:
        sev = None

    # meta/emotion → plain dict (numpy → python) → Json(...)
    emotion_plain = _to_jsonable(emotion)
    meta_plain = _to_jsonable(meta or {})

    # face_id: если в таблице нет отдельной колонки — положим в meta
    if face_id is not None:
        meta_plain.setdefault("face_id", face_id)

    # user_id / device_id к uuid
    dev_uuid = _as_uuid(device_id)
    usr_uuid = _as_uuid(user_id)

    # lat/lon к float
    lat_f = _as_float(lat)
    lon_f = _as_float(lon)

    # ---- 3) Работаем с БД
    db: Session = SessionLocal()
    try:
        # выясняем, какие колонки существуют на самом деле
        cols_res = db.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = 'alerts'
            """)
        )
        existing = {row[0] for row in cols_res}

        data: Dict[str, Any] = {}

        # id: если колонка есть — зададим сами (совместимо и с default-ами)
        if "id" in existing:
            data["id"] = uuid.uuid4()

        # время: captured_at -> created_at -> ts_utc
        if "captured_at" in existing:
            data["captured_at"] = now
        elif "created_at" in existing:
            data["created_at"] = now
        elif "ts_utc" in existing:
            data["ts_utc"] = now

        # связи/атрибуты
        if "device_id" in existing:
            data["device_id"] = dev_uuid
        if "user_id" in existing:
            data["user_id"] = usr_uuid
        if "label" in existing:
            data["label"] = label
        if "severity" in existing and sev is not None:
            data["severity"] = sev
        if "face_id" in existing:
            data["face_id"] = face_id

        if "lat" in existing:
            data["lat"] = lat_f
        if "lon" in existing:
            data["lon"] = lon_f
        if "address" in existing:
            data["address"] = address
        if "zone" in existing:
            data["zone"] = zone

        # json поля
        if "emotion" in existing:
            data["emotion"] = Json(emotion_plain)
        if "meta" in existing:
            data["meta"] = Json(meta_plain)

        # пути (поддержка двух схем имён) — СТРОГО относительные без "alerts/"
        if "raw_path" in existing:
            data["raw_path"] = raw_rel
        elif "image_path" in existing:
            data["image_path"] = raw_rel

        if "face_path" in existing:
            data["face_path"] = face_rel
        elif "image_face_path" in existing:
            data["image_face_path"] = face_rel

        # формируем INSERT только по реально существующим колонкам
        columns = list(data.keys())
        placeholders = [f":{k}" for k in columns]
        sql = text(
            f"INSERT INTO alerts ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"RETURNING id"
        )
        new_id = db.execute(sql, data).scalar()
        db.commit()

        # что вернуть фронту
        returned_face_id = face_id or meta_plain.get("face_id")
        return {
            "id": str(new_id) if new_id else (str(data.get("id")) if data.get("id") else None),
            "face_id": returned_face_id,
            # можно оставить для внутренних задач
            "raw_path": raw_rel,
            "face_path": face_rel,
            # фронт использует эти URL
            "raw_url": raw_url,
            "face_url": face_url,
        }
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()
