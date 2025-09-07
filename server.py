from fastapi import FastAPI, WebSocket
from fastapi.concurrency import run_in_threadpool
import base64
import io
import numpy as np
from PIL import Image
import cv2
import asyncio
import logging
from pathlib import Path
import time
import uuid
from typing import Optional, Dict, Any
import os
from datetime import datetime, timezone

from face_recognizer_insightface import FaceRecognizer
from app.services.alert_persist import create_alert_record
from emotion_recognizer import analyze_emotions
from face_emotion_buffer import FaceTracker  # Импортируем буфер эмоций
from app.middleware.ws_auth import TokenWebSocketAuthMiddleware

# -------------------- ЛОГИ --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_processor")

# -------------------- НАСТРОЙКИ --------------------
from app.core.config_env import settings
DEBUG_SAVE_FRAMES = bool(str(settings.DEBUG_SAVE_FRAMES).lower() in ("1", "true", "yes", "on"))
ALERT_NOTIFY_THRESHOLD = int(settings.ALERT_NOTIFY_THRESHOLD)

app = FastAPI()

app.add_middleware(
    TokenWebSocketAuthMiddleware,
    protected_paths={"/ws/alerts"},    # можно расширять множеством путей
    # allowed_roles по умолчанию {"admin", "officer"}; см. app/middleware/ws_auth.py
)

face_recognizer = FaceRecognizer()
emotion_tracker = FaceTracker()  # Инициализируем трекер эмоций

# API (RBAC/ingest/alerts) смонтированы через общий роутер
from app.api.v1.router import api_router as rbac_api_router
app.include_router(rbac_api_router, prefix="/api/v1")
from app.api.v1.routes import admin_users, admin_devices  
app.include_router(admin_users.router)                     
app.include_router(admin_devices.router)                   

# -------------------- УТИЛИТЫ --------------------
def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    else:
        return obj

def _guess_ext(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8\xff"):   # JPEG
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        return ".png"
    return ".bin"

def _strip_data_url_prefix(b64txt: str) -> str:
    # убираем 'data:image/jpeg;base64,' если прилетает data URL
    if ";base64," in b64txt:
        return b64txt.split(";base64,", 1)[1]
    return b64txt

def save_debug_frame(raw_bytes: bytes, src: str = "ws") -> Optional[str]:
    """
    Сохраняет кадр на диск: debug_frames/<src>/<YYYYMMDD>/<HHMMSS_ms_uuid>.<ext>
    Возвращает путь или None при ошибке.
    Работает только если DEBUG_SAVE_FRAMES включён.
    """
    if not DEBUG_SAVE_FRAMES:
        return None
    try:
        day = time.strftime("%Y%m%d")
        out_dir = Path("debug_frames") / src / day
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%H%M%S")
        ms = int((time.time() % 1) * 1000)
        ext = _guess_ext(raw_bytes)
        fname = f"{ts}_{ms:03d}_{uuid.uuid4().hex[:8]}{ext}"
        fpath = out_dir / fname
        with open(fpath, "wb") as f:
            f.write(raw_bytes)
        return str(fpath)
    except Exception as e:
        logger.exception("Failed to save debug frame: %s", e)
        return None

def _pil_from_raw(raw_bytes: bytes) -> Image.Image:
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return pil_img

def _as_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# -------------------- МЕНЕДЖЕР УВЕДОМЛЕНИЙ ПО WS --------------------
class AlertsWSManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.info("Alerts subscriber connected; total=%d", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        logger.info("Alerts subscriber disconnected; total=%d", len(self._connections))

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        if not self._connections:
            return
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

alerts_ws_manager = AlertsWSManager()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def maybe_notify_alert(source: str, face_id: str, emotion: Dict[str, Any], meta: Optional[Dict[str, Any]] = None):
    """
    Если агрессия >= порога — рассылаем событие всем подписчикам на /ws/alerts.
    """
    try:
        score = float(emotion.get("aggression_score", 0))
    except Exception:
        score = 0.0
    severity = int(round(score))

    if severity >= ALERT_NOTIFY_THRESHOLD:
        payload = {
            "event": "alert",
            "ts": _now_iso(),
            "source": source,               # "ws" | "ws_stream"
            "face_id": face_id,
            "aggression_score": score,
            "severity": severity,
            "emotion": emotion,
            "meta": meta or {},
        }
        await alerts_ws_manager.broadcast(payload)


# -------------------- ПРЕЖНИЕ ВСПОМОГАТЕЛЬНЫЕ --------------------
def is_valid_face_result(result):
    if isinstance(result, list):
        return len(result) > 0
    elif isinstance(result, np.ndarray):
        return result.size > 0
    return False

known_faces = {}  # Хранилище эмбеддингов для идентификации лиц (пока не используется)


# -------------------- ОБЩАЯ ОБРАБОТКА КАДРА --------------------
async def _process_frame_and_collect_result(raw_bytes: bytes, source: str) -> dict:
    """
    Вызов конвейера: распознавание лица -> буфер -> эмоции.
    Возвращает dict {face_id: {emotion, profile}} или {}.
    Параллельно — отправка уведомлений при превышении порога.
    """
    # распознавание лица (совместимо с FaceRecognizer.process_image)
    face_result = await run_in_threadpool(face_recognizer.process_image, raw_bytes)
    if face_result != []:
        for face_data in face_result:
            embedding = face_data.get("embedding")
            face_img = face_data.get("face")   # обрезанное лицо (BGR)
            face_id = face_data.get("face_id")
            if embedding is None or face_img is None or face_id is None:
                continue
            emotion_tracker.update_face(face_id, face_img)

    # проверяем готовые буферы и анализируем эмоции
    ready_faces = emotion_tracker.get_ready_buffers()
    result = {}
    for face_id, best_img in ready_faces:
        emotion = await run_in_threadpool(analyze_emotions, best_img)
        if isinstance(emotion, dict):
            try:
                score = float(emotion.get("aggression_score", 0))
            except Exception:
                score = 0.0
            if score > 1:  # тот же порог, что в emotion_recognizer.py
                create_alert_record(
                    original_bytes=raw_bytes,      # полный кадр
                    face_bgr=best_img,             # кроп лица
                    device_id=None,                # если появится device_id — подставим
                    face_id=face_id,
                    emotion=emotion,
                    meta={"source": source},       # можно добавить всё, что нужно
                    lat=None, lon=None, address=None,
                    label="aggression",
                )
        logger.info("Emotion for %s: %s", face_id, emotion)
        result[face_id] = {
            "emotion": emotion,
            "profile": "anime"
        }
        # уведомления
        if isinstance(emotion, dict):
            await maybe_notify_alert(source=source, face_id=face_id, emotion=emotion)

    if result:
        # очищаем после успешной отправки
        emotion_tracker.cleanup()
    else:
        logger.warning("Лицо не найдено или нераспознано.")

    return to_serializable(result)


# -------------------- ВЕБСОКЕТ: BASE64-ТЕКСТ (/ws) --------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Ожидает текстовые сообщения — base64 изображения (JPEG/PNG).
    При DEBUG_SAVE_FRAMES=1 сохраняет каждый кадр в debug_frames/ws/YYYYMMDD/.
    """
    await websocket.accept()
    while True:
        try:
            data_b64 = await websocket.receive_text()
            data_b64 = _strip_data_url_prefix(data_b64)
            img_data = base64.b64decode(data_b64)

            saved_path = save_debug_frame(img_data, src="ws")
            if saved_path:
                logger.info("WS frame saved: %s (len=%d bytes)", saved_path, len(img_data))

            # sanity-check декодирования (не критично)
            try:
                pil_img = _pil_from_raw(img_data)
                logger.info("WS decode: %sx%s", pil_img.width, pil_img.height)
            except Exception as dec_e:
                logger.warning("WS decode failed: %s", dec_e)

            result = await _process_frame_and_collect_result(img_data, source="ws")
            if result:
                await websocket.send_json(result)

        except Exception as e:
            logger.error("Ошибка при обработке кадра (ws): %s", e, exc_info=True)
            try:
                if 'img_data' in locals():
                    save_debug_frame(img_data, src="ws_error")
            except Exception:
                pass
            await websocket.send_json({"error": str(e)})
            await asyncio.sleep(0.1)


# -------------------- ВЕБСОКЕТ: СТРИМ БАЙТОВ/BASE64 (/ws/stream) --------------------
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Поддерживает 2 формата входа:
      - binary: сырой JPEG/PNG байтовый кадр
      - text: base64 изображения
    При DEBUG_SAVE_FRAMES=1 сохраняет кадры в debug_frames/ws_stream/YYYYMMDD/.
    """
    await websocket.accept()
    while True:
        try:
            msg = await websocket.receive()
            # msg: {'type': 'websocket.receive', 'text': '...', 'bytes': b'...'}
            if "bytes" in msg and msg["bytes"] is not None:
                img_data = msg["bytes"]
            elif "text" in msg and msg["text"] is not None:
                data_b64 = _strip_data_url_prefix(msg["text"])
                img_data = base64.b64decode(data_b64)
            else:
                # пустое сообщение — пропускаем
                continue

            saved_path = save_debug_frame(img_data, src="ws_stream")
            if saved_path:
                logger.info("WS_STREAM frame saved: %s (len=%d bytes)", saved_path, len(img_data))

            # sanity-check
            try:
                pil_img = _pil_from_raw(img_data)
                logger.info("WS_STREAM decode: %sx%s", pil_img.width, pil_img.height)
            except Exception as dec_e:
                logger.warning("WS_STREAM decode failed: %s", dec_e)

            result = await _process_frame_and_collect_result(img_data, source="ws_stream")
            if result:
                await websocket.send_json(result)

        except Exception as e:
            logger.error("Ошибка при обработке кадра (ws_stream): %s", e, exc_info=True)
            try:
                if 'img_data' in locals():
                    save_debug_frame(img_data, src="ws_stream_error")
            except Exception:
                pass
            await websocket.send_json({"error": str(e)})
            await asyncio.sleep(0.1)


# -------------------- КАНАЛ УВЕДОМЛЕНИЙ ПО WS --------------------
@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    """
    Подписка на уведомления об алертах.
    Сервер отправляет события {"event":"alert", "ts":..., "face_id":..., "aggression_score":..., "severity":...}
    когда агрессия >= ALERT_NOTIFY_THRESHOLD.
    """
    await alerts_ws_manager.connect(ws)
    try:
        # держим соединение; читаем/игнорируем входящие пинги клиента
        while True:
            try:
                _ = await ws.receive_text()
            except Exception:
                # клиент мог отправить небинарное, игнорируем
                await asyncio.sleep(0.2)
    except Exception:
        pass
    finally:
        alerts_ws_manager.disconnect(ws)


# -------------------- СИСТЕМНЫЕ ЭНДПОИНТЫ --------------------
@app.on_event("shutdown")
def shutdown_event():
    face_recognizer.close()

@app.get("/")
async def root():
    return {"message": "Бэкенд работает"}
