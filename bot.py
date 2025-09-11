import telebot
from config import TELEGRAM_TOKEN, TELEGRAM_FACES_CHAT_ID
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
import io
from app.models.system_settings import SystemSettings
from PIL import Image
import cv2
from requests import RequestException
import logging
import numpy as np

bot = telebot.TeleBot(TELEGRAM_TOKEN)

def telegram_enabled(db: Session | None = None) -> bool:
    own = False
    if db is None:
        db = SessionLocal()
        own = True
    try:
        st = db.query(SystemSettings).get(1)
        return bool(getattr(st, "telegram_enabled", True)) if st else True
    finally:
        if own:
            db.close()

def send_aggression_alert(image_np, aggression_score):
    if not telegram_enabled():
        return  
    try:
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
    except RequestException as e:
        logging.warning("telegram send failed (network): %s", e)
    except Exception as e:
        logging.warning("telegram send failed: %s", e)
