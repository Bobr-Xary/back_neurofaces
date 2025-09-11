# app/api/v1/ingest.py
import base64, io
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from PIL import Image

from app.db.session import get_db
from app.core.deps import get_current_device
from app.schemas.alert import AlertIngestRequest

router = APIRouter(prefix="/ingest", tags=["ingest"])

def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None

def _b64_to_bytes(image_b64: str, _strip_data_url_prefix) -> bytes:
    try:
        b64 = _strip_data_url_prefix(image_b64)
        raw = base64.b64decode(b64)
        # Лёгкая валидация формата
        _ = Image.open(io.BytesIO(raw)).convert("RGB")
        return raw
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image_b64")

@router.post("/frame")
async def ingest_frame(
    payload: AlertIngestRequest,
    db: Session = Depends(get_db),
    device = Depends(get_current_device),
    flush: bool = Query(False, description="Форс-дренаж буфера как при закрытии WS"),
):
    """
    REST-инжест кадра. Поведение идентично WS:
    1) Декодируем кадр, кладём лицо(+raw) в общий буфер (тот же, что использует WS).
    2) Фоновый воркер создаёт alert и отправляет WS-уведомление.
    3) (опционально) flush=1 — форс-дренаж буфера прямо сейчас.
    """
    # ЛЕНИВЫЙ ИМПОРТ из server.py — разрывает цикл импорта
    from server import (
        _process_frame_and_collect_result_enriched as _ws_like_process,
        _flush_pending_buffers as _ws_like_flush,
        _strip_data_url_prefix,
        save_debug_frame,
    )

    raw_bytes = _b64_to_bytes(payload.image_b64, _strip_data_url_prefix)
    save_debug_frame(raw_bytes, src="rest_ingest")

    # Контекст, как у WS: device_id, владелец как user_id, geo и meta
    device_id = str(device.id)
    user_id = str(device.owner_user_id) if getattr(device, "owner_user_id", None) else None

    # Гео может быть на верхнем уровне или внутри location
    lat = _to_float(getattr(payload, "lat", None))
    lon = _to_float(getattr(payload, "lon", None))
    address = getattr(payload, "address", None)

    if (lat is None or lon is None) and getattr(payload, "location", None):
        loc = payload.location or {}
        lat = lat if lat is not None else _to_float(loc.get("lat"))
        lon = lon if lon is not None else _to_float(loc.get("lon"))
        if address is None:
            address = loc.get("address")

    meta: Dict[str, Any] = getattr(payload, "meta", None) or {}

    # ТА САМАЯ логика, что в WS: кладём кадр в буфер
    await _ws_like_process(
        raw_bytes,
        source="rest_ingest",
        device_id=device_id,
        user_id=user_id,
        lat=lat, lon=lon, address=address,
        extra_meta=meta,
    )

    # По запросу — моментально «досчитать» и записать алёрты (аналог disconnect у WS)
    if flush:
        await _ws_like_flush(
            source="rest_ingest",
            device_id=device_id, user_id=user_id,
            lat=lat, lon=lon, address=address,
            extra_meta=meta,
        )
        return {"status": "queued", "flushed": True}

    return {"status": "queued"}
