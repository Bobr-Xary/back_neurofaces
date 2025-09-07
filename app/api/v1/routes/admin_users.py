from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List
from uuid import UUID
# TODO: adjust import if needed
from sqlalchemy.orm import Session
from app.api.v1.deps_admin import get_db, admin_required
from app.models.user import User

router = APIRouter(prefix="/api/v1/admin/users", tags=["admin:users"])

@router.get("", response_model=List[dict])
def list_users(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _admin = Depends(admin_required),
):
    q = db.query(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
    items = q.all()
    # Пример простой сериализации; при наличии Pydantic-схем — замените.
    return [
        {
            "id": str(u.id),
            "email": getattr(u, "email", None),
            "role": getattr(u, "role", None) or getattr(u, "role_name", None),
            "created_at": getattr(u, "created_at", None).isoformat() if getattr(u, "created_at", None) else None,
            "is_active": getattr(u, "is_active", True),
        }
        for u in items
    ]

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: UUID,
    db: Session = Depends(get_db),
    _admin = Depends(admin_required),
):
    inst = db.query(User).filter(User.id == user_id).first()
    if not inst:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(inst)
    db.commit()
    return
