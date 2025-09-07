
import os
import pathlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

import sys
sys.path.append(os.getcwd())

from app.api.v1.router import api_router
from app.db.session import SessionLocal, engine
from app.models.user import User
from app.models.enums import UserRole
from app.core.security import hash_password

MIGRATIONS = [
    "migrations/sql/0001_init.sql",
    "migrations/sql/0002_alerts.sql",
]

@pytest.fixture(scope="session", autouse=True)
def check_env():
    db_url = os.getenv("DATABASE_URL")
    assert db_url, "DATABASE_URL env var must be set for tests (e.g., postgresql+psycopg2://postgres:postgres@localhost:5432/postgres)"
    os.environ.setdefault("JWT_SECRET", "TEST_JWT_SECRET_CHANGE_ME")
    os.environ.setdefault("JWT_ALG", "HS256")

@pytest.fixture(scope="session")
def apply_migrations(check_env):
    for path in MIGRATIONS:
        p = pathlib.Path(path)
        assert p.exists(), f"Migration file not found: {p}"
        sql = p.read_text(encoding="utf-8")
        with engine.begin() as conn:
            conn.execute(text(sql))

@pytest.fixture(scope="function")
def db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(scope="session")
def app_instance(apply_migrations) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    return app

@pytest.fixture(scope="function")
def client(app_instance) -> TestClient:
    with TestClient(app_instance) as c:
        yield c

@pytest.fixture(scope="function")
def ensure_admin(db: Session):
    email = "admin@example.com"
    admin = db.query(User).filter(User.email == email).first()
    if not admin:
        admin = User(
            email=email,
            full_name="Admin",
            password_hash=hash_password("test"),
            role=UserRole.admin,
            is_active=True,
        )
        db.add(admin); db.commit()
    yield {"email": email, "password": "test"}
