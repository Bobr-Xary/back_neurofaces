from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.deps import get_current_user
from app.db.session import get_db

router = APIRouter(prefix="/media", tags=["media"])

BASE = Path("alerts").resolve()

def _role_str(user) -> str:
    role_raw = getattr(user, "role", None)
    role_str = getattr(role_raw, "value", role_raw)
    return (str(role_str or "")).lower()

@router.get("/alerts/{path:path}")
def get_alert_media(
    path: str,
    user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Нормализуем путь и защищаемся от traversal
    rel = Path(path).as_posix()
    full = (BASE / rel).resolve()
    if not str(full).startswith(str(BASE)):
        raise HTTPException(status_code=404, detail="Not found")

    # Находим алерт по пути (raw/face)
    row = db.execute(text("""
        SELECT a.id, a.device_id, a.user_id AS creator_id, a.hidden,
               a.raw_path, a.face_path,
               d.owner_user_id AS owner_user_id
          FROM alerts a
          LEFT JOIN devices d ON d.id = a.device_id
         WHERE a.raw_path = :rel OR a.face_path = :rel
         LIMIT 1
    """), {"rel": rel}).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    is_face = (row.get("face_path") == rel)
    is_raw  = (row.get("raw_path")  == rel)

    role = _role_str(user)
    uid = str(user.id)

    # Админ — всегда можно
    if role == "admin":
        pass
    else:
        # Скрытые нельзя
        if bool(row.get("hidden")):
            raise HTTPException(status_code=403, detail="Forbidden")

        is_owner   = (row.get("owner_user_id") and str(row["owner_user_id"]) == uid)
        is_creator = (row.get("creator_id")    and str(row["creator_id"])    == uid)

        # ACL (может быть None)
        acl = db.execute(text("""
            SELECT can_view_face
              FROM alert_access
             WHERE alert_id = :aid AND user_id = :uid
             LIMIT 1
        """), {"aid": row["id"], "uid": uid}).mappings().first()
        acl_can_face = bool(acl["can_view_face"]) if acl and "can_view_face" in acl else False

        if role == "officer":
            # Видеть файл можно, если офицер связан с алертом
            if not (is_owner or is_creator or acl):
                raise HTTPException(status_code=403, detail="Forbidden")
            # Для лица — если не владелец и не создатель, то нужен явный can_view_face
            if is_face and not (is_owner or is_creator or acl_can_face):
                raise HTTPException(status_code=403, detail="Forbidden")

        elif role in ("citizen", "user"):
            # Гражданским raw не отдаём никогда
            if is_raw:
                raise HTTPException(status_code=403, detail="Forbidden")
            # Лица — только при явном разрешении
            if not (is_face and acl_can_face):
                raise HTTPException(status_code=403, detail="Forbidden")
        else:
            raise HTTPException(status_code=403, detail="Forbidden")

    if not full.is_file():
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(str(full))
