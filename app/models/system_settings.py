
from datetime import datetime
from sqlalchemy import Column, Integer, Boolean, String, DateTime
from app.db.session import Base

class SystemSettings(Base):
    __tablename__ = "system_settings"
    id = Column(Integer, primary_key=True, default=1)
    telegram_enabled = Column(Boolean, nullable=False, default=True)
    telegram_chat_id = Column(String(255), nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
