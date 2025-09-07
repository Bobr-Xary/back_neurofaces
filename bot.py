import telebot
from config import TELEGRAM_TOKEN, TELEGRAM_FACES_CHAT_ID
from datetime import datetime
import io
from PIL import Image
import cv2
import numpy as np

bot = telebot.TeleBot(TELEGRAM_TOKEN)

def send_aggression_alert(image_np, aggression_score):
    # Преобразуем NumPy-изображение в байты
    image_rgb = image_np[:, :, ::-1]  # BGR → RGB
    pil_img = Image.fromarray(image_rgb)
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Подпись
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    caption = f"⚠️ Агрессивный человек\nВремя: {timestamp}\nОценка агрессии: {aggression_score:.2f}"

    bot.send_photo(chat_id=TELEGRAM_FACES_CHAT_ID, photo=img_bytes, caption=caption)

