
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional

def grant_alert_access(db: Session, alert_id: str, user_id: Optional[str], can_view_face: bool):
    if not user_id:
        return
    sql = text("""
        INSERT INTO alert_access (alert_id, user_id, can_view_face)
        VALUES (:aid, :uid, :face)
        ON CONFLICT (alert_id, user_id) DO UPDATE SET can_view_face = EXCLUDED.can_view_face
    """)
    db.execute(sql, {"aid": alert_id, "uid": user_id, "face": bool(can_view_face)})
