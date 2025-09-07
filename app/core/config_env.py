from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"

    # Auth / JWT
    JWT_SECRET: str = "CHANGE_ME_DEV_SECRET"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS (CSV в .env: http://localhost:3000,http://127.0.0.1:3000)
    CORS_ORIGINS: List[str] = []

    # Alerts
    ALERT_NOTIFY_THRESHOLD: int = 6

    # Telegram
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID_ADMIN: Optional[str] = None

    # Admin API key (optional legacy)
    ADMIN_API_KEY: Optional[str] = None

    # Debug
    DEBUG_SAVE_FRAMES: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_cors(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

settings = Settings()
