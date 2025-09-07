from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List
from uuid import UUID
# TODO: adjust import if needed
from sqlalchemy.orm import Session
from app.api.v1.deps_admin import get_db, admin_required
from app.models.device import Device

router = APIRouter(prefix="/api/v1/admin/devices", tags=["admin:devices"])

@router.get("", response_model=List[dict])
def list_devices(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _admin = Depends(admin_required),
):
    q = db.query(Device).order_by(Device.created_at.desc()).limit(limit).offset(offset)
    items = q.all()
    return [
        {
            "id": str(d.id),
            "name": getattr(d, "name", None),
            "uid": getattr(d, "uid", None) or getattr(d, "device_uid", None),
            "created_at": getattr(d, "created_at", None).isoformat() if getattr(d, "created_at", None) else None,
            "is_active": getattr(d, "is_active", True),
            "last_seen_at": getattr(d, "last_seen_at", None).isoformat() if getattr(d, "last_seen_at", None) else None,
            "location": {
                "lat": getattr(d, "lat", None),
                "lon": getattr(d, "lon", None),
            },
        }
        for d in items
    ]

@router.delete("/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_device(
    device_id: UUID,
    db: Session = Depends(get_db),
    _admin = Depends(admin_required),
):
    inst = db.query(Device).filter(Device.id == device_id).first()
    if not inst:
        raise HTTPException(status_code=404, detail="Device not found")
    db.delete(inst)
    db.commit()
    return
