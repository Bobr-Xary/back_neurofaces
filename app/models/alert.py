import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.db.session import Base

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ts_utc = Column(DateTime, nullable=False, default=datetime.utcnow)
    device_id = Column(UUID(as_uuid=True), ForeignKey("devices.id", ondelete="SET NULL"), nullable=True, index=True)
    severity = Column(Integer, nullable=False, default=0)  # 0..10
    label = Column(String(64), nullable=False, default="aggression")
    face_id = Column(String(64), nullable=True)  # text to avoid FK to legacy faces table
    emotion = Column(JSONB, nullable=True)      # DeepFace/FER results
    meta = Column(JSONB, nullable=True)         # any extra fields
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    address = Column(String(255), nullable=True)
    image_path = Column(Text, nullable=True)         # original snapshot
    image_face_path = Column(Text, nullable=True)    # cropped face (if available)
