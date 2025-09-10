
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.deps import get_current_user, require_role
from app.models.enums import UserRole
from app.models.system_settings import SystemSettings
from app.schemas.settings import SystemSettingsOut, SystemSettingsUpdate

router = APIRouter(prefix="/admin", tags=["admin-settings"])

@router.get("/settings", response_model=SystemSettingsOut)
def get_settings(user=Depends(get_current_user), db: Session = Depends(get_db)):
    require_role(user, {UserRole.admin})
    st = db.query(SystemSettings).get(1)
    if not st:
        st = SystemSettings(id=1, telegram_enabled=True)
        db.add(st); db.commit(); db.refresh(st)
    return SystemSettingsOut(telegram_enabled=st.telegram_enabled, telegram_chat_id=st.telegram_chat_id)

@router.put("/settings", response_model=SystemSettingsOut)
def update_settings(payload: SystemSettingsUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    require_role(user, {UserRole.admin})
    st = db.query(SystemSettings).get(1)
    if not st:
        st = SystemSettings(id=1, telegram_enabled=True)
        db.add(st)
    if payload.telegram_enabled is not None:
        st.telegram_enabled = bool(payload.telegram_enabled)
    if payload.telegram_chat_id is not None:
        st.telegram_chat_id = payload.telegram_chat_id
    db.commit(); db.refresh(st)
    return SystemSettingsOut(telegram_enabled=st.telegram_enabled, telegram_chat_id=st.telegram_chat_id)
