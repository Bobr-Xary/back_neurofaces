
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from app.core.deps import get_current_user, require_role
from app.models.enums import UserRole

router = APIRouter(prefix="/media", tags=["media"])

BASE = Path("alerts").resolve()

@router.get("/alerts/{path:path}")
def get_alert_media(path: str, user=Depends(get_current_user)):
    # Only officers and admins may fetch photos
    require_role(user, {UserRole.admin, UserRole.officer})
    full = (BASE / path).resolve()
    if not str(full).startswith(str(BASE)):
        raise HTTPException(status_code=404, detail="Not found")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(full))
