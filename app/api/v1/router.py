# app/api/v1/router.py
from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.devices import router as devices_router

# добавить ЭТИ:
from app.api.v1.ingest import router as ingest_router
from app.api.v1.alerts import router as alerts_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(devices_router)

# и ЭТИ:
api_router.include_router(ingest_router)
api_router.include_router(alerts_router)
