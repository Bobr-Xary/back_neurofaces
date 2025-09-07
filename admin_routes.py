# admin_routes.py
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from config import POSTGRES_CONFIG, ADMIN_API_KEY

router = APIRouter(prefix="/admin", tags=["admin"])

def get_conn():
    return psycopg2.connect(
        host=POSTGRES_CONFIG['host'],
        port=POSTGRES_CONFIG['port'],
        dbname=POSTGRES_CONFIG['dbname'],
        user=POSTGRES_CONFIG['user'],
        password=POSTGRES_CONFIG['password']
    )

def require_admin(x_admin_token: str = Header(..., alias="X-Admin-Token")):
    if x_admin_token != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

def init_admin_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Пользователи
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Девайсы
            cur.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    id SERIAL PRIMARY KEY,
                    device_uid TEXT UNIQUE NOT NULL,
                    name TEXT,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    lat DOUBLE PRECISION,
                    lon DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP
                );
            """)
        conn.commit()

class DeviceOut(BaseModel):
    id: int
    device_uid: str
    name: Optional[str] = None
    user_id: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    created_at: Optional[str] = None
    last_seen: Optional[str] = None

@router.on_event("startup")
def _startup():
    # Безопасно создаём таблицы если их нет
    init_admin_tables()

@router.delete("/users/{user_id}")
def delete_user(user_id: int, _: str = require_admin()):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE id=%s", (user_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="User not found")
            cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
        conn.commit()
    return {"status": "ok", "deleted_user_id": user_id}

@router.get("/devices", response_model=List[DeviceOut])
def list_devices(
    _: str = require_admin(),
    owner_user_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None, description="Поиск по UID/имени"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    sql = "SELECT id, device_uid, name, user_id, lat, lon, created_at, last_seen FROM devices"
    where = []
    params: List[Any] = []
    if owner_user_id is not None:
        where.append("user_id = %s")
        params.append(owner_user_id)
    if search:
        where.append("(device_uid ILIKE %s OR name ILIKE %s)")
        params.extend([f"%{search}%", f"%{search}%"])
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return rows

@router.delete("/devices/{id_or_uid}")
def delete_device(id_or_uid: str, _: str = require_admin()):
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Пытаемся как id (int), иначе как uid (text)
            as_id = None
            try:
                as_id = int(id_or_uid)
            except ValueError:
                pass

            if as_id is not None:
                cur.execute("DELETE FROM devices WHERE id=%s RETURNING id", (as_id,))
            else:
                cur.execute("DELETE FROM devices WHERE device_uid=%s RETURNING id", (id_or_uid,))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"status": "ok", "deleted_device_id": row[0]}
