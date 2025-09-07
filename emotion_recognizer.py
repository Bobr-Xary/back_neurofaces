import numpy as np
import threading
import logging
from deepface import DeepFace
from fer import FER
import mediapipe as mp
from PIL import Image
import io
from datetime import datetime
import cv2
import os
from io import BytesIO
from gfpgan import GFPGANer
from basicsr.utils import imwrite
from bot import send_aggression_alert


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Инициализация MediaPipe для анализа ключевых точек лица
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Инициализация FER для распознавания эмоций
emotion_detector = FER()

# Загрузка модели GFPGAN
restorer = GFPGANer(
        model_path='GFPGAN/gfpgan/weights/GFPGANv1.3.pth',
        upscale=3,
        arch='clean',  # Для версий '1.3' или '1.4'
        channel_multiplier=2,
        bg_upsampler=None  # Если фоновый апскейлинг не используется
    )

def preprocess_image(image):
    """
    Выполняет предобработку изображения:
      - Изменение размера до стандартного формата (например, 224x224) для улучшения качества анализа.
      - Коррекция яркости/контраста, если необходимо.
      - Преобразование из BGR в RGB.
    """
    # Преобразуем изображение из BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Изменяем размер изображения (опционально – можно экспериментировать с различными размерами)
    image_resized = cv2.resize(image_rgb, (224, 224))
    return image_resized

def get_face_landmarks(image):
    """
    Извлекает ключевые точки лица с использованием MediaPipe и масштабирует их по размеру изображения.
    """
    height, width, _ = image.shape  # Получаем размеры изображения
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Обрабатываем изображение с указанием его размеров
    results = face_mesh.process(image_rgb)
    
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # Масштабируем нормализованные координаты обратно на размер изображения
                landmarks.append((landmark.x * width, landmark.y * height, landmark.z))  # Масштабируем по ширине и высоте
    return landmarks

def preprocess_for_fer(image):
    """
    Дополнительная предобработка изображения для улучшения качества анализа FER.
    """
    # Конвертируем изображение в цветное (если оно в оттенках серого)
    if len(image.shape) == 2:  # Если изображение в оттенках серого
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Увеличиваем контрастность (если нужно)
    alpha = 1.5  # Контраст
    beta = 20    # Яркость
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image




def analyze_emotions(image_bytes):
    try:
        # Если это массив — убедимся, что он C-contiguous
        if isinstance(image_bytes, np.ndarray):
            if not image_bytes.flags['C_CONTIGUOUS']:
                image_bytes = np.ascontiguousarray(image_bytes)
            image = image_bytes
        else:
            pil_img = Image.open(io.BytesIO(image_bytes))
            image = np.array(pil_img)

        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        processed_image = preprocess_image(image)

        # 1. Анализ эмоций через DeepFace
        result = DeepFace.analyze(processed_image, actions=['emotion'], enforce_detection=False)
        emotions = result.get("emotion", {}) if isinstance(result, dict) else result[0].get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "") if isinstance(result, dict) else result[0].get("dominant_emotion", "")

        # 2. Анализ эмоций через FER
        
        preprocessed_image = preprocess_for_fer(image)
        #upscale = upscale_with_gfpgan(preprocessed_image)
        fer_emotion_main, _ = emotion_detector.top_emotion(preprocessed_image)
        fer_emotions = emotion_detector.detect_emotions(preprocessed_image)  # Извлекаем эмоцию из FER

        # 3. Анализ ключевых точек лица через MediaPipe
        landmarks = get_face_landmarks(image)  # Получаем ключевые точки

        # 4. Вычисление агрессии по микромимике
        features = extract_maximal_emotion_features(landmarks)
        aggression_score = estimate_aggression_score_from_maximal_features(features, dominant_emotion)

        # 5. Отправляем событие в телеграмм, если опасный человек найден
        if aggression_score > 1:
            # Преобразуем обратно в NumPy для отправки
            threading.Thread(target=send_aggression_alert, args=(image, aggression_score)).start()

        # Возвращаем результат анализа
        timestamp = datetime.utcnow().isoformat()
        return {
            "dominant_emotion": dominant_emotion,
            #"all_emotions": emotions,
            "aggression_score":aggression_score,
            "fer_emotions": fer_emotion_main,
            #"landmarks": landmarks,  # Добавим ключевые точки
            #"timestamp": timestamp
        }

    except Exception as e:
        logging.exception("Ошибка анализа эмоций: %s", str(e))
        return {"error": str(e)}




def upscale_with_gfpgan(image_ndarray):
    """Интерфейс для апскейлинга изображений с использованием GFPGAN"""

    try:
        # Преобразуем NumPy массив (BGR) в RGB
        input_img = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB для GFPGAN

        # Применяем восстановление
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=False,  # Убедитесь, что изображения не выровнены
            only_center_face=False,
            paste_back=True
        )

        # Преобразуем восстановленное изображение обратно в формат NumPy (BGR)
        restored_img_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)  # Преобразуем в BGR для OpenCV

        # Преобразуем изображение в байты, если необходимо (например, для WebSocket)
        result_image = cv2.cvtColor(restored_img_bgr, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB для корректного отображения
        pil_img = Image.fromarray(result_image)  # Преобразуем в формат PIL
        img_byte_arr = BytesIO()  # Используем BytesIO для преобразования в байты
        pil_img.save(img_byte_arr, format='PNG')  # Сохраняем в формате PNG
        img_byte_arr = img_byte_arr.getvalue()  # Получаем байтовое представление изображения

        save_restore_path = os.path.join("out", "puk.png")
        with open(save_restore_path, 'wb') as f:
            f.write(img_byte_arr)  # Сохраняем изображение в байтах

        return restored_img_bgr  # Возвращаем изображение в формате NumPy ndarray

    except Exception as e:
        logging.exception("Ошибка при апскейлинге изображения: %s", str(e))
        return {"error": str(e)}

# =========================
# AGGRESSION METRICS SECTION
# =========================
aggression_scores = []
aggression_timestamps = []

def extract_maximal_emotion_features(landmarks):
    """
    Расширенный (максимальный) набор признаков для анализа эмоций, включающий ~60+ характеристик.
    Использует координаты 468 точек MediaPipe.
    """
    if not landmarks or len(landmarks) < 468:
        return {}

    def dist(p1, p2):
        return np.linalg.norm(np.array(landmarks[p1][:2]) - np.array(landmarks[p2][:2]))

    def vertical(p): return landmarks[p][1]
    def horizontal(p): return landmarks[p][0]

    features = {}

    # --- Брови (внутренние, внешние) ---
    features['brow_inner_distance'] = dist(66, 296)
    features['brow_outer_distance'] = dist(105, 334)
    features['brow_left_height'] = vertical(66) - vertical(10)
    features['brow_right_height'] = vertical(296) - vertical(10)
    features['brow_height_avg'] = (features['brow_left_height'] + features['brow_right_height']) / 2
    features['brow_height_diff'] = abs(features['brow_left_height'] - features['brow_right_height'])

    # --- Глаза (левый и правый) ---
    eye_pts = {
        "left": (159, 145, 33, 133),
        "right": (386, 374, 362, 263)
    }
    for side, (top, bottom, left, right) in eye_pts.items():
        features[f'eye_{side}_height'] = dist(top, bottom)
        features[f'eye_{side}_width'] = dist(left, right)
        features[f'eye_{side}_aspect'] = features[f'eye_{side}_height'] / max(features[f'eye_{side}_width'], 1)
        features[f'eye_{side}_center_y'] = (vertical(top) + vertical(bottom)) / 2

    features['eye_aspect_ratio_diff'] = abs(features['eye_left_aspect'] - features['eye_right_aspect'])
    features['eye_height_diff'] = abs(features['eye_left_height'] - features['eye_right_height'])

    # --- Нос ---
    features['nose_bridge_length'] = dist(168, 2)
    features['nose_tip_to_lip'] = dist(2, 13)
    features['nose_tip_to_chin'] = dist(2, 152)
    features['nose_width'] = dist(94, 331)

    # --- Рот ---
    features['mouth_width'] = dist(61, 291)
    features['mouth_height'] = dist(13, 14)
    features['mouth_aspect_ratio'] = features['mouth_height'] / max(features['mouth_width'], 1)
    features['mouth_corner_slope'] = np.degrees(np.arctan2(vertical(61) - vertical(291), horizontal(61) - horizontal(291)))
    features['mouth_asymmetry'] = abs(vertical(61) - vertical(291))

    # --- Улыбка / оскал ---
    features['lip_upper_to_nose'] = dist(2, 13)
    features['lip_lower_to_chin'] = dist(14, 152)
    features['lip_ratio'] = features['lip_upper_to_nose'] / max(features['lip_lower_to_chin'], 1)

    # --- Щёки ---
    features['cheek_to_eye_left'] = dist(234, 159)
    features['cheek_to_eye_right'] = dist(454, 386)
    features['cheek_raise'] = (features['cheek_to_eye_left'] + features['cheek_to_eye_right']) / 2
    features['cheek_asymmetry'] = abs(features['cheek_to_eye_left'] - features['cheek_to_eye_right'])

    # --- Подбородок и лоб ---
    features['chin_to_lip'] = dist(152, 14)
    features['forehead_to_brow_center'] = vertical(10) - ((vertical(66) + vertical(296)) / 2)
    features['face_height'] = dist(10, 152)
    features['face_width'] = dist(234, 454)

    # --- Симметрия ---
    symmetry_pairs = [(33, 263), (159, 386), (234, 454), (61, 291), (105, 334)]
    symmetry_diffs = [abs(vertical(i) - vertical(j)) for i, j in symmetry_pairs]
    features['vertical_symmetry_avg'] = 1.0 - np.clip(np.mean(symmetry_diffs) / 15.0, 0, 1)
    features['vertical_symmetry_std'] = np.std(symmetry_diffs)

    # --- Голова ---
    left_eye = np.array([horizontal(33), vertical(33)])
    right_eye = np.array([horizontal(263), vertical(263)])
    dx, dy = right_eye - left_eye
    features['head_tilt_angle'] = np.degrees(np.arctan2(dy, dx))

    # --- Отношения ---
    features['eye_to_mouth_ratio'] = (features['eye_left_height'] + features['eye_right_height']) / (2 * max(features['mouth_height'], 1))
    features['brow_to_eye_ratio'] = features['brow_inner_distance'] / max((features['eye_left_height'] + features['eye_right_height']) / 2, 1)
    features['mouth_to_nose_ratio'] = features['mouth_height'] / max(features['nose_tip_to_lip'], 1)

    return features


def estimate_aggression_score_from_maximal_features(features, dominant_emotion):
    """
    Расчёт уровня агрессии (0–10) по расширенным признакам лица.
    Чем выше итоговый скор — тем выше вероятность агрессии.
    """

    score = 0.0

    # --- Брови ---
    if features['brow_inner_distance'] < 25:
        score += 1.0  # напряжённость
    if features['brow_height_diff'] > 3:
        score += 0.5  # асимметрия
    if abs(features['brow_outer_distance'] - features['brow_inner_distance']) < 10:
        score += 0.3  # агрессивное сжатие

    # --- Глаза ---
    if features['eye_left_height'] < 3 or features['eye_right_height'] < 3:
        score += 1.0  # прищур
    if features['eye_aspect_ratio_diff'] > 0.3:
        score += 0.5  # асимметрия глаз
    if features['eye_height_diff'] > 1.0:
        score += 0.5
    if features['eye_to_mouth_ratio'] < 0.6:
        score += 0.5  # маленькие глаза, большой рот — индикатор напряжения

    # --- Рот ---
    if features['mouth_aspect_ratio'] < 0.15:
        score += 1.0  # сжатие губ
    elif features['mouth_aspect_ratio'] > 0.6:
        score += 0.5  # оскал или крик
    if features['mouth_asymmetry'] > 3:
        score += 0.5
    if features['mouth_corner_slope'] < -5:  # уголки вниз
        score += 0.5
    if features['mouth_corner_slope'] > 5:  # уголки вверх — нейтрально
        score -= 0.3

    # --- Щёки и нос ---
    if features['cheek_raise'] < 10:
        score += 0.5
    if features['lip_upper_to_nose'] < 7:
        score += 0.8  # оскал
    if features['nose_width'] > 30:
        score += 0.3  # напряжённые крылья носа

    # --- Подбородок и лоб ---
    if features['chin_to_lip'] > 25:
        score += 0.3  # напряжённость
    if features['forehead_to_brow_center'] < 10:
        score += 0.2  # брови опущены

    # --- Симметрия ---
    if features['vertical_symmetry_avg'] < 0.85:
        score += 1.0  # сильная асимметрия
    elif features['vertical_symmetry_avg'] > 0.95:
        score -= 0.3  # расслабленность

    # --- Наклон головы ---
    if abs(features['head_tilt_angle']) > 15:
        score += 0.5
    elif abs(features['head_tilt_angle']) < 5:
        score -= 0.2  # прямая осанка — контроль

    # --- Соотношения ---
    if features['brow_to_eye_ratio'] > 2.5:
        score += 0.5
    if features['mouth_to_nose_ratio'] > 1.0:
        score += 0.3

    # Нормализация
    final_score = round(np.clip(score, 0, 10), 2)
    if dominant_emotion == "angry":
        final_score += 2
    return final_score


    

def log_aggression_score(landmarks):
    score = estimate_aggression_score_from_maximal_features(landmarks)
    timestamp = datetime.utcnow()
    aggression_scores.append(score)
    aggression_timestamps.append(timestamp)
    plot_aggression()
    return score


def plot_aggression():
    import matplotlib.pyplot as plt
    if not aggression_scores:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(aggression_timestamps, aggression_scores, label='Aggression Score', color='red', marker='o')
    plt.title('Aggression Score Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Aggression Level (0-10)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("out", exist_ok=True)
    plt.savefig("out/aggression_plot.png")
    plt.close()

