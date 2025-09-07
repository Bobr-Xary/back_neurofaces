import cv2
import numpy as np
import datetime
import os
from collections import defaultdict

class FaceEmotionBuffer:
    def __init__(self, face_id):
        self.face_id = face_id
        self.frames = []  # Список кадров: (время, изображение, оценка качества)
        self.start_time = datetime.datetime.now()  # Время создания буфера

    def add_frame(self, image):
        ts = datetime.datetime.now()  # Текущее время
        score = self.compute_quality(image)  # Оценка качества кадра
        self.frames.append((ts, image, score))  # Добавляем кадр в буфер

    def compute_quality(self, image):
        # Переводим изображение в серый цвет
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Оцениваем резкость через дисперсию Лапласиана
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Средняя яркость
        brightness = np.mean(gray)
        # Итоговая оценка: 50% резкость + 50% яркость
        return 0.5 * sharpness + 0.5 * brightness

    def get_best_frame(self):
        # Если буфер пуст, возвращаем None
        if not self.frames:
            return None
        # Возвращаем изображение с максимальной оценкой качества
        return max(self.frames, key=lambda x: x[2])[1]

    def is_ready(self, max_age=5):
        # Проверка, прошло ли max_age секунд с начала буфера
        now = datetime.datetime.now()
        return (now - self.start_time).total_seconds() >= max_age

class FaceTracker:
    def __init__(self, buffer_timeout=5, idle_forget_time=5):
        self.buffers = {}  # face_id -> FaceEmotionBuffer
        self.last_seen = {}  # face_id -> время последнего появления
        self.buffer_timeout = buffer_timeout  # Таймаут буфера, после которого он считается готовым
        self.idle_forget_time = idle_forget_time  # Время, через которое забываются неактивные лица

        # Папка для отладки, сохраняющая лучшие кадры
        self.debug_folder = "debug_best_frames"
        os.makedirs(self.debug_folder, exist_ok=True)

    def update_face(self, face_id, image):
        now = datetime.datetime.now()

        # Если лицо новое или давно не появлялось — создаем новый буфер
        if face_id not in self.buffers or (now - self.last_seen[face_id]).total_seconds() > self.idle_forget_time:
            self.buffers[face_id] = FaceEmotionBuffer(face_id)

        # Добавляем кадр в соответствующий буфер
        self.buffers[face_id].add_frame(image)
        # Обновляем время последнего появления
        self.last_seen[face_id] = now

    def get_ready_buffers(self):
        ready = []
        for face_id, buf in self.buffers.items():
            # Проверяем, готов ли буфер к анализу
            if buf.is_ready(max_age=self.buffer_timeout):
                best = buf.get_best_frame()
                if best is not None:
                    ready.append((face_id, best))  # Добавляем лучший кадр в список

                    # Сохраняем лучший кадр для отладки
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"face_{face_id}_{timestamp}.jpg"
                    filepath = os.path.join(self.debug_folder, filename)
                    cv2.imwrite(filepath, best)

        return ready

    def cleanup(self):
        to_delete = []
        now = datetime.datetime.now()
        for face_id, buf in self.buffers.items():
            # Получаем время последнего появления или время создания буфера
            last_seen_time = self.last_seen.get(face_id, buf.start_time)
            # Если лицо давно не появлялось — помечаем на удаление
            if (now - last_seen_time).total_seconds() > self.idle_forget_time:
                to_delete.append(face_id)

        # Удаляем устаревшие буферы
        for face_id in to_delete:
            self.buffers.pop(face_id, None)
            self.last_seen.pop(face_id, None)
