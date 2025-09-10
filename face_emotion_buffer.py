import cv2
import numpy as np
import datetime
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# --------- helpers ---------
def _valid_img(img) -> bool:
    return isinstance(img, np.ndarray) and img.size > 0 and img.ndim in (2, 3)

def _score_img(image: np.ndarray) -> float:
    if not _valid_img(image):
        return -1.0
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        # Итоговая оценка: 50% резкость + 50% яркость
        return 0.5 * float(sharpness) + 0.5 * brightness
    except Exception:
        return -1.0

@dataclass
class FaceSample:
    ts: datetime.datetime
    face_bgr: np.ndarray
    score: float
    raw_bytes: bytes  # исходный кадр (тот же, из которого сделан кроп)

# --------- буфер одного лица ---------
class FaceEmotionBuffer:
    def __init__(self, face_id: str):
        self.face_id = face_id
        self.frames: List[FaceSample] = []
        self.start_time = datetime.datetime.now()

    def add_frame(self, image: np.ndarray, raw_bytes: Optional[bytes] = None, frame_ts: Optional[datetime.datetime] = None):
        if not _valid_img(image):
            return
        ts = frame_ts or datetime.datetime.now()
        score = _score_img(image)
        self.frames.append(FaceSample(ts=ts, face_bgr=image, score=score, raw_bytes=raw_bytes or b""))

    def compute_quality(self, image):
        # Оставлено для обратной совместимости с внешними вызовами
        return _score_img(image)

    def get_best_frame(self) -> Optional[np.ndarray]:
        """Старое поведение: вернуть ЛУЧШЕЕ изображение (кроп) — для обратной совместимости."""
        if not self.frames:
            return None
        best = max(self.frames, key=lambda s: s.score)
        return best.face_bgr

    def get_best_sample(self) -> Optional[FaceSample]:
        """Новое поведение: вернуть лучший сэмпл (кроп + raw_bytes + ts)."""
        if not self.frames:
            return None
        return max(self.frames, key=lambda s: s.score)

    def is_ready(self, max_age=5):
        now = datetime.datetime.now()
        return (now - self.start_time).total_seconds() >= max_age

# --------- трекер лиц ---------
class FaceTracker:
    def __init__(self, buffer_timeout=5, idle_forget_time=5):
        self.buffers: Dict[str, FaceEmotionBuffer] = {}   # face_id -> FaceEmotionBuffer
        self.last_seen: Dict[str, datetime.datetime] = {} # face_id -> dt
        self.buffer_timeout = buffer_timeout
        self.idle_forget_time = idle_forget_time

        # Папка для отладки, сохраняющая лучшие кропы
        self.debug_folder = "debug_best_frames"
        os.makedirs(self.debug_folder, exist_ok=True)

    def update_face(self, face_id: str, image: np.ndarray,
                    raw_bytes: Optional[bytes] = None,
                    frame_ts: Optional[float] = None):
        """
        Обновляем буфер лица.
        raw_bytes — исходный кадр, из которого сделан кроп 'image'.
        frame_ts  — timestamp (секунды) исходного кадра; если None — возьмём сейчас.
        """
        now = datetime.datetime.now()
        # если лицо новое/давно не появлялось — создаём новый буфер
        if face_id not in self.buffers or (now - self.last_seen.get(face_id, now)).total_seconds() > self.idle_forget_time:
            self.buffers[face_id] = FaceEmotionBuffer(face_id)

        # преобразуем ts
        ts_dt = datetime.datetime.fromtimestamp(frame_ts, tz=None) if isinstance(frame_ts, (int, float)) else now

        # добавляем кадр
        self.buffers[face_id].add_frame(image=image, raw_bytes=raw_bytes, frame_ts=ts_dt)
        # обновляем last_seen
        self.last_seen[face_id] = now

    def get_ready_buffers(self) -> List[Tuple[str, np.ndarray]]:
        """
        СТАРОЕ API: вернуть список (face_id, best_crop).
        Оставлено для обратной совместимости.
        """
        ready: List[Tuple[str, np.ndarray]] = []
        for face_id, buf in self.buffers.items():
            if buf.is_ready(max_age=self.buffer_timeout):
                best_img = buf.get_best_frame()
                if _valid_img(best_img):
                    ready.append((face_id, best_img))
                    # debug save
                    try:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"face_{face_id}_{timestamp}.jpg"
                        filepath = os.path.join(self.debug_folder, filename)
                        cv2.imwrite(filepath, best_img)
                    except Exception:
                        pass
        return ready

    def get_all_best_samples(self):
        """
        Вернуть лучший сэмпл по каждому face_id БЕЗ ожидания timeout:
        [(face_id, best_face_bgr, best_raw_bytes, ts_unix), ...]
        """
        out = []
        for face_id, buf in self.buffers.items():
            best = buf.get_best_sample() if hasattr(buf, "get_best_sample") else None
            if best is None:
                # совместимость со старой структурой frames: (ts, image, score)
                try:
                    if buf.frames:
                        ts, img, _ = max(buf.frames, key=lambda x: x[2])
                        out.append((face_id, img, b"", ts.timestamp() if hasattr(ts, "timestamp") else None))
                except Exception:
                    continue
            else:
                out.append((face_id, best.face_bgr, best.raw_bytes or b"", best.ts.timestamp()))
        return out

    def get_ready_buffers_enriched(self) -> List[Tuple[str, np.ndarray, bytes, float]]:
        """
        НОВОЕ API: вернуть список (face_id, best_crop, best_raw_bytes, ts_unix).
        Это нужно, чтобы сохранять кроп и «общий план» из одного и того же кадра.
        """
        out: List[Tuple[str, np.ndarray, bytes, float]] = []
        for face_id, buf in self.buffers.items():
            if buf.is_ready(max_age=self.buffer_timeout):
                best = buf.get_best_sample()
                if not best or not _valid_img(best.face_bgr):
                    continue
                ts_unix = best.ts.timestamp()
                out.append((face_id, best.face_bgr, best.raw_bytes or b"", ts_unix))
                # debug save кропа
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"face_{face_id}_{timestamp}.jpg"
                    filepath = os.path.join(self.debug_folder, filename)
                    cv2.imwrite(filepath, best.face_bgr)
                except Exception:
                    pass
        return out

    def cleanup(self):
        to_delete = []
        now = datetime.datetime.now()
        for face_id, buf in self.buffers.items():
            last_seen_time = self.last_seen.get(face_id, buf.start_time)
            if (now - last_seen_time).total_seconds() > self.idle_forget_time:
                to_delete.append(face_id)
        for face_id in to_delete:
            self.buffers.pop(face_id, None)
            self.last_seen.pop(face_id, None)
