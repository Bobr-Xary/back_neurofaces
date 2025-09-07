from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.deps import get_current_user
from app.models.alert import Alert
from app.models.enums import UserRole
from app.schemas.alert import AlertOut

router = APIRouter(prefix="/alerts", tags=["alerts"])

@router.get("/", response_model=List[AlertOut])
def list_alerts(
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
):
    q = db.query(Alert).order_by(Alert.ts_utc.desc())
    if user.role == UserRole.admin:
        pass
    elif user.role == UserRole.officer:
        from app.models.device import Device
        device_ids = [d.id for d in db.query(Device).filter(Device.owner_user_id == user.id).all()]
        if device_ids:
            q = q.filter(Alert.device_id.in_(device_ids))
        else:
            q = q.filter(False)
    elif user.role == UserRole.citizen:
        q = q.filter(False)  # citizens will get an anon feed later
    else:
        q = q.filter(False)
    return q.limit(limit).all()
