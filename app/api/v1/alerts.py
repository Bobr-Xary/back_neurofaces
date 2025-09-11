from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text, bindparam
import json

from app.api.v1.deps_admin import get_db, get_current_user
from app.models.enums import UserRole
from app.models.device import Device  # для фильтра офицеров

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _media_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if path.startswith("http") or path.startswith("/"):
        return path
    # теперь раздаём под /alerts/*
    return f"/api/v1/media/alerts/{path}"


def _row_get(row_map, *names):
    """Безопасно получить первое существующее поле из row._mapping."""
    for n in names:
        if n in row_map:
            return row_map[n]
    return None


def _meta_as_dict(val) -> dict:
    """meta может прийти как dict или как JSON-строка — нормализуем в dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return {}


def _face_id_from_row(row_map) -> Optional[str]:
    """face_id из колонки или из meta.face_id (фолбэк)."""
    fid = _row_get(row_map, "face_id")
    if fid:
        return str(fid)
    meta = _meta_as_dict(_row_get(row_map, "meta"))
    fid = meta.get("face_id")
    return str(fid) if fid else None


@router.get("/")
def list_alerts(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Возвращает последние алерты.
      - admin: все
      - officer: только алерты его устройств
      - citizen/прочие: пока пусто
    Поддерживает как старые поля (image_path/image_face_path, created_at),
    так и новые (raw_path/face_path, captured_at).
    """
    base_sql = "SELECT * FROM alerts"
    params = {"limit": limit}

    if user.role == UserRole.admin:
        final_sql = f"{base_sql} ORDER BY id DESC LIMIT :limit"
        rows = db.execute(text(final_sql), params).mappings().all()

    elif user.role == UserRole.officer:
        device_ids = [
            d.id for d in db.query(Device).filter(Device.owner_user_id == user.id).all()
        ]
        if not device_ids:
            return []
        sql = text(
            "SELECT * FROM alerts WHERE device_id IN :ids ORDER BY id DESC LIMIT :limit"
        ).bindparams(bindparam("ids", expanding=True))
        rows = db.execute(sql, {"ids": device_ids, "limit": limit}).mappings().all()

    else:
        return []

    items = []
    for r in rows:
        ts = _row_get(r, "captured_at", "created_at")  # ts_utc может не существовать
        raw_rel = _row_get(r, "raw_path", "image_path")
        face_rel = _row_get(r, "face_path", "image_face_path")
        meta_dict = _meta_as_dict(_row_get(r, "meta"))

        items.append(
            {
                "id": str(r["id"]) if "id" in r else None,
                "ts_utc": ts.isoformat() if ts is not None else None,
                "device_id": (
                    str(r["device_id"])
                    if "device_id" in r and r["device_id"] is not None
                    else None
                ),
                "user_id": (
                    str(r["user_id"])
                    if "user_id" in r and r["user_id"] is not None
                    else None
                ),
                "face_id": _face_id_from_row(r),
                "severity": _row_get(r, "severity"),
                "label": _row_get(r, "label"),
                "emotion": _row_get(r, "emotion"),
                "meta": meta_dict or None,
                "raw_url": _media_url(raw_rel),
                "face_url": _media_url(face_rel),
                "lat": _row_get(r, "lat"),
                "lon": _row_get(r, "lon"),
                "address": _row_get(r, "address"),
                "zone": _row_get(r, "zone"),
            }
        )
    return items


@router.get("/{alert_id}")
def get_alert_detail(
    alert_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """
    Детали алерта по id, с fallback по именам столбцов:
      - время: captured_at | created_at
      - пути: raw_path|image_path, face_path|image_face_path
      - face_id: отдельная колонка или meta.face_id
    """
    row = db.execute(
        text("SELECT * FROM alerts WHERE id = :id"), {"id": alert_id}
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")

    ts = _row_get(row, "captured_at", "created_at")
    raw_rel = _row_get(row, "raw_path", "image_path")
    face_rel = _row_get(row, "face_path", "image_face_path")
    meta_dict = _meta_as_dict(_row_get(row, "meta"))

    return {
        "id": str(row["id"]) if "id" in row else None,
        "captured_at": ts.isoformat() if ts is not None else None,
        "device_id": (
            str(row["device_id"])
            if "device_id" in row and row["device_id"] is not None
            else None
        ),
        "user_id": (
            str(row["user_id"])
            if "user_id" in row and row["user_id"] is not None
            else None
        ),
        "face_id": _face_id_from_row(row),
        "lat": _row_get(row, "lat"),
        "lon": _row_get(row, "lon"),
        "address": _row_get(row, "address"),
        "zone": _row_get(row, "zone"),
        "emotion": _row_get(row, "emotion"),
        "meta": meta_dict or {},
        "raw_path": raw_rel,
        "face_path": face_rel,
        "raw_url": _media_url(raw_rel),
        "face_url": _media_url(face_rel),
    }
