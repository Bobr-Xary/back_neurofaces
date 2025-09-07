import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Enum, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.db.session import Base
from app.models.enums import DeviceType

class Device(Base):
    __tablename__ = "devices"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(Enum(DeviceType, name="device_type", native_enum=False), nullable=False)
    # Only store hashed secret (bcrypt/argon recommended). Plaintext is only shown at creation time.
    secret_hash = Column(String(255), nullable=False)
    owner_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, nullable=True)

    # Optional static location metadata (can be overridden by frame payloads)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    zone = Column(String(255), nullable=True)
