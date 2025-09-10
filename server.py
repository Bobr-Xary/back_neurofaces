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
from jose import jwt, JWTError
import uuid
from typing import Optional, Dict, Any
import os
from datetime import datetime, timezone
from starlette.websockets import WebSocketDisconnect, WebSocketState

from starlette.staticfiles import StaticFiles
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

ALERTS_DIR = Path(os.getenv("ALERTS_DIR", "alerts")).resolve()
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

app.add_middleware(
    TokenWebSocketAuthMiddleware,
    protected_paths={"/ws/alerts"},    # можно расширять множеством путей
    # allowed_roles по умолчанию {"admin", "officer"}; см. app/middleware/ws_auth.py
)
app.mount("/alerts", StaticFiles(directory=str(ALERTS_DIR)), name="alerts-static")

face_recognizer = FaceRecognizer()
emotion_tracker = FaceTracker()  # Инициализируем трекер эмоций
emotion_buf_lock = asyncio.Lock()
_background_face_task: asyncio.Task | None = None


# API (RBAC/ingest/alerts) смонтированы через общий роутер
from app.api.v1.router import api_router as rbac_api_router
app.include_router(rbac_api_router, prefix="/api/v1")
from app.api.v1.routes import admin_users, admin_devices  
app.include_router(admin_users.router)                     
app.include_router(admin_devices.router)        
           
@app.on_event("startup")
async def on_startup():
    global _background_face_task
    if _background_face_task is None:
        _background_face_task = asyncio.create_task(_background_face_worker())
        logger.info("Background face worker started")

@app.on_event("shutdown")
def shutdown_event():
    face_recognizer.close()
    # остановим фонового воркера
    global _background_face_task
    try:
        if _background_face_task:
            _background_face_task.cancel()
    except Exception:
        pass



# -------------------- УТИЛИТЫ --------------------
async def _background_face_worker():
    """
    Периодически вынимаем из FaceTracker готовые пары (crop+raw),
    считаем эмоции, создаём алерты и шлём уведомления — без привязки к WS.
    """
    CHECK_INTERVAL_SEC = 0.5  # как часто проверять буфер
    while True:
        try:
            # читаем и чистим буфер атомарно
            async with emotion_buf_lock:
                ready = []
                if hasattr(emotion_tracker, "get_ready_buffers_enriched"):
                    ready = emotion_tracker.get_ready_buffers_enriched()
                elif hasattr(emotion_tracker, "get_ready_buffers"):
                    # старый API (без raw): оборачиваем
                    ready_simple = emotion_tracker.get_ready_buffers()
                    ready = [(fid, img, b"", time.time()) for fid, img in ready_simple]

            if ready:
                for face_id, best_img, best_raw, ts_sec in ready:
                    # считаем эмоции вне локов
                    emotion = await run_in_threadpool(analyze_emotions, best_img)
                    if not isinstance(emotion, dict):
                        continue
                    try:
                        score = float(emotion.get("aggression_score", 0))
                    except Exception:
                        score = 0.0
                    if score > 1:
                        captured_dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc) if ts_sec else None
                        alert_info = create_alert_record(
                            original_bytes=best_raw or b"",
                            face_bgr=best_img,
                            device_id=None, user_id=None,
                            face_id=face_id,
                            emotion=emotion,
                            meta={"source": "bg_worker"},
                            lat=None, lon=None, address=None,
                            label="aggression",
                            captured_at=captured_dt,
                        )
                        await maybe_notify_alert(
                            source="bg_worker", face_id=face_id, emotion=emotion,
                            alert_id=alert_info.get("id") if alert_info else None,
                            raw_url=alert_info.get("raw_url") if alert_info else None,
                            face_url=alert_info.get("face_url") if alert_info else None,
                            meta=None,
                        )
                # после обработки — подчистка буфера (если ваш FaceTracker не чистит точечно)
                async with emotion_buf_lock:
                    emotion_tracker.cleanup()
        except Exception:
            logger.exception("background face worker failed")
        await asyncio.sleep(CHECK_INTERVAL_SEC)

async def _ws_send_json_safe(ws: WebSocket, payload: Dict[str, Any]) -> None:
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(payload)
    except Exception:
        pass

def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    else:
        return obj

async def _flush_pending_buffers(
    *,
    source: str,
    device_id: Optional[str],
    user_id: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    address: Optional[str],
    extra_meta: Optional[Dict[str, Any]],
):
    """
    Форс-обработка текущего содержимого буфера эмоций (без ожидания timeout).
    Создаёт алерты и шлёт WS-уведомления так же, как в основном пайплайне.
    """
    # забираем лучшие сэмплы по каждому face_id
    if hasattr(emotion_tracker, "get_all_best_samples"):
        samples = emotion_tracker.get_all_best_samples()
    elif hasattr(emotion_tracker, "get_ready_buffers_enriched"):
        samples = emotion_tracker.get_ready_buffers_enriched()
    else:
        samples = []

    if not samples:
        return

    for item in samples:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            face_id, best_img, best_raw, ts_sec = item
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            face_id, best_img = item[0], item[1]
            best_raw, ts_sec = b"", time.time()
        else:
            continue

        emotion = await run_in_threadpool(analyze_emotions, best_img)

        alert_info = None
        if isinstance(emotion, dict):
            try:
                score = float(emotion.get("aggression_score", 0))
            except Exception:
                score = 0.0
            if score > 1:
                captured_dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc) if ts_sec else None
                alert_info = create_alert_record(
                    original_bytes=best_raw or b"",  # best_raw уже из того же кадра
                    face_bgr=best_img,
                    device_id=device_id,
                    user_id=user_id,
                    face_id=face_id,
                    emotion=emotion,
                    meta={"source": source, **(extra_meta or {})} if extra_meta else {"source": source},
                    lat=lat, lon=lon, address=address,
                    label="aggression",
                    captured_at=captured_dt,
                )
                await maybe_notify_alert(
                    source=source, face_id=face_id, emotion=emotion,
                    alert_id=alert_info.get("id") if alert_info else None,
                    raw_url=alert_info.get("raw_url") if alert_info else None,
                    face_url=alert_info.get("face_url") if alert_info else None,
                    meta=extra_meta,
                )

    # очищаем буфер после форс-обработки
    emotion_tracker.cleanup()

def _guess_ext(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8\xff"):   # JPEG
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        return ".png"
    return ".bin"

def _decode_token(token: Optional[str]) -> Optional[dict]:
    if not token: return None
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        return None
    
async def _process_frame_and_collect_result_enriched(
    raw_bytes: bytes, source: str,
    device_id: Optional[str] = None, user_id: Optional[str] = None,
    lat: Optional[float] = None, lon: Optional[float] = None, address: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None
) -> dict:
    face_result = await run_in_threadpool(face_recognizer.process_image, raw_bytes)
    if face_result != []:
        for face_data in face_result:
            embedding = face_data.get("embedding")
            face_img = face_data.get("face")   # BGR
            face_id = face_data.get("face_id")
            if embedding is None or face_img is None or face_id is None:
                continue
            emotion_tracker.update_face(face_id, face_img, raw_bytes=raw_bytes, frame_ts=time.time())

    ready_faces = emotion_tracker.get_ready_buffers_enriched()
    result = {}
    for face_id, best_img, best_raw, ts_sec in ready_faces:
        emotion = await run_in_threadpool(analyze_emotions, best_img)
        alert_info = None
        if isinstance(emotion, dict):
            try:
                score = float(emotion.get("aggression_score", 0))
            except Exception:
                score = 0.0
            if score > 1:
                captured_dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc) if ts_sec else None
                alert_info = create_alert_record(
                    original_bytes=best_raw or raw_bytes,   # ВАЖНО: raw из того же сэмпла, что и кроп
                    face_bgr=best_img,
                    device_id=device_id,
                    user_id=user_id,
                    face_id=face_id,
                    emotion=emotion,
                    meta={"source": source, **(extra_meta or {})},
                    lat=lat, lon=lon, address=address,
                    label="aggression",
                    captured_at=captured_dt,
                )
        logger.info("Emotion for %s: %s", face_id, emotion)
        result[face_id] = {"emotion": emotion, "profile": "anime"}
        # уведомления (с alert_id и ссылками если есть)
        if isinstance(emotion, dict):
            await maybe_notify_alert(
                source=source, face_id=face_id, emotion=emotion,
                alert_id=alert_info.get("id") if alert_info else None,
                raw_url=alert_info.get("raw_url") if alert_info else None,
                face_url=alert_info.get("face_url") if alert_info else None,
                meta=extra_meta,
            )

    if result: emotion_tracker.cleanup()
    else: logger.warning("Лицо не найдено или нераспознано.")
    return to_serializable(result)

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


async def maybe_notify_alert(source: str, face_id: str, emotion: Dict[str, Any],
                             alert_id: Optional[str] = None,
                             raw_url: Optional[str] = None,
                             face_url: Optional[str] = None,
                             meta: Optional[Dict[str, Any]] = None):
    try:
        score = float(emotion.get("aggression_score", 0))
    except Exception:
        score = 0.0
    severity = int(round(score))
    if severity >= ALERT_NOTIFY_THRESHOLD:
        payload = {
            "event": "alert",
            "ts": _now_iso(),
            "source": source,
            "alert_id": alert_id,
            "face_id": face_id,
            "aggression_score": score,
            "severity": severity,
            "emotion": emotion,
            "raw_url": raw_url,
            "face_url": face_url,
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
            emotion_tracker.update_face(face_id, face_img, raw_bytes=raw_bytes, frame_ts=time.time())


    # проверяем готовые буферы и анализируем эмоции
    ready_faces = emotion_tracker.get_ready_buffers_enriched()
    result = {}
    for face_id, best_img, best_raw, ts_sec in ready_faces:
        emotion = await run_in_threadpool(analyze_emotions, best_img)
        emotion = await run_in_threadpool(analyze_emotions, best_img)

        alert_info = None
        if isinstance(emotion, dict):
            try:
                score = float(emotion.get("aggression_score", 0))
            except Exception:
                score = 0.0
            if score > 1:
                captured_dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc) if ts_sec else None
                alert_info = create_alert_record(
                    original_bytes=best_raw or raw_bytes,   # ВАЖНО: raw из того же сэмпла
                    face_bgr=best_img,
                    device_id=None,
                    user_id=None,
                    face_id=face_id,
                    emotion=emotion,
                    meta={"source": source},
                    lat=None, lon=None, address=None,
                    label="aggression",
                    captured_at=captured_dt,
                )

        logger.info("Emotion for %s: %s", face_id, emotion)
        result[face_id] = {
            "emotion": emotion,
            "profile": "anime"
        }

        # уведомления (если был создан alert)
        if isinstance(emotion, dict):
            await maybe_notify_alert(
                source=source, face_id=face_id, emotion=emotion,
                alert_id=alert_info.get("id") if alert_info else None,
                raw_url=alert_info.get("raw_url") if alert_info else None,
                face_url=alert_info.get("face_url") if alert_info else None,
                meta=None,
            )

    if result:
        # очищаем после успешной отправки
        emotion_tracker.cleanup()
    else:
        logger.warning("Лицо не найдено или нераспознано.")

    return to_serializable(result)


# -------------------- ВЕБСОКЕТ: BASE64-ТЕКСТ (/ws) --------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    token = websocket.query_params.get("token")
    claims = _decode_token(token)
    user_id = claims.get("sub") if claims else None

    # держим последние известные метаданные, чтобы использовать при дренажe
    last_lat = last_lon = None
    last_address = None
    last_device_id = None
    last_meta: Dict[str, Any] | None = None

    while True:
        if websocket.client_state != WebSocketState.CONNECTED:
            logger.info("WS state=%s -> break", websocket.client_state)
            break
        try:
            data = await websocket.receive_text()

            meta = {}
            if data and data.strip().startswith("{"):
                import json
                obj = json.loads(data)
                data_b64 = _strip_data_url_prefix(obj.get("image_b64", ""))
                meta = obj.get("meta", {})
                lat = obj.get("lat"); lon = obj.get("lon"); address = obj.get("address")
                device_id = obj.get("device_id")
            else:
                data_b64 = _strip_data_url_prefix(data)
                lat = lon = address = None
                device_id = None

            # обновим «последние» метаданные
            last_lat = lat if lat is not None else last_lat
            last_lon = lon if lon is not None else last_lon
            last_address = address if address is not None else last_address
            last_device_id = device_id if device_id is not None else last_device_id
            last_meta = meta or last_meta

            img_data = base64.b64decode(data_b64)
            save_debug_frame(img_data, src="ws")
            try:
                pil_img = _pil_from_raw(img_data)
                logger.info("WS decode: %sx%s", pil_img.width, pil_img.height)
            except Exception as dec_e:
                logger.warning("WS decode failed: %s", dec_e)

            result = await _process_frame_and_collect_result_enriched(
                img_data, source="ws",
                device_id=last_device_id, user_id=user_id,
                lat=last_lat, lon=last_lon, address=last_address, extra_meta=last_meta
            )
            if result:
                await _ws_send_json_safe(websocket, result)

        except WebSocketDisconnect as e:
            logger.info("WS disconnected: code=%s reason=%r", getattr(e, "code", None), getattr(e, "reason", None))
            # ФОРС-ДРЕНАЖ: дообрабатываем то, что накопили
            await _flush_pending_buffers(
                source="ws",
                device_id=last_device_id, user_id=user_id,
                lat=last_lat, lon=last_lon, address=last_address,
                extra_meta=last_meta,
            )
            break
        except RuntimeError as e:
            logger.info("WS runtime closed: %s", e)
            await _flush_pending_buffers(
                source="ws",
                device_id=last_device_id, user_id=user_id,
                lat=last_lat, lon=last_lon, address=last_address,
                extra_meta=last_meta,
            )
            break
        except Exception as e:
            logger.exception("Ошибка при обработке кадра (ws): %s", e)
            try:
                if 'img_data' in locals():
                    save_debug_frame(img_data, src="ws_error")
            except Exception:
                pass
            await _ws_send_json_safe(websocket, {"error": str(e)})
            await asyncio.sleep(0.1)




# -------------------- ВЕБСОКЕТ: СТРИМ БАЙТОВ/BASE64 (/ws/stream) --------------------
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    # сюда обычно девайсы не присылают geo, но оставим «последние» поля на будущее
    last_lat = last_lon = None
    last_address = None
    last_device_id = None
    last_meta: Dict[str, Any] | None = None

    while True:
        if websocket.client_state != WebSocketState.CONNECTED:
            logger.info("WS_STREAM state=%s -> break", websocket.client_state)
            break
        try:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                img_data = msg["bytes"]
            elif "text" in msg and msg["text"] is not None:
                data_b64 = _strip_data_url_prefix(msg["text"])
                img_data = base64.b64decode(data_b64)
            else:
                continue

            save_debug_frame(img_data, src="ws_stream")
            try:
                pil_img = _pil_from_raw(img_data)
                logger.info("WS_STREAM decode: %sx%s", pil_img.width, pil_img.height)
            except Exception as dec_e:
                logger.warning("WS_STREAM decode failed: %s", dec_e)

            result = await _process_frame_and_collect_result_enriched(
                img_data, source="ws_stream",
                device_id=last_device_id, user_id=None,
                lat=last_lat, lon=last_lon, address=last_address, extra_meta=last_meta
            )
            if result:
                await _ws_send_json_safe(websocket, result)

        except WebSocketDisconnect as e:
            logger.info("WS_STREAM disconnected: code=%s reason=%r", getattr(e, "code", None), getattr(e, "reason", None))
            await _flush_pending_buffers(
                source="ws_stream",
                device_id=last_device_id, user_id=None,
                lat=last_lat, lon=last_lon, address=last_address,
                extra_meta=last_meta,
            )
            break
        except RuntimeError as e:
            logger.info("WS_STREAM runtime closed: %s", e)
            await _flush_pending_buffers(
                source="ws_stream",
                device_id=last_device_id, user_id=None,
                lat=last_lat, lon=last_lon, address=last_address,
                extra_meta=last_meta,
            )
            break
        except Exception as e:
            logger.exception("Ошибка при обработке кадра (ws_stream): %s", e)
            try:
                if 'img_data' in locals():
                    save_debug_frame(img_data, src="ws_stream_error")
            except Exception:
                pass
            await _ws_send_json_safe(websocket, {"error": str(e)})
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
