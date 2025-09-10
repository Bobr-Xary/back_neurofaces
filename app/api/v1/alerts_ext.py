
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db
from app.core.deps import get_current_user, require_role
from app.models.enums import UserRole
from app.schemas.alerts_ext import ShareAlertRequest, HideAlertRequest

router = APIRouter(prefix="/alerts", tags=["alerts-ext"])

def _time_expr():
    return "COALESCE(a.captured_at, a.created_at, a.ts_utc)"

@router.get("", response_model=List[Dict[str, Any]])
def list_alerts(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=500),
    since: Optional[str] = Query(None, description="ISO datetime; return alerts since this time"),
    include_hidden: bool = Query(False, description="Admin-only: include hidden alerts"),
):
    uid = str(user.id)
    params = {"uid": uid, "limit": limit}
    time_filter = ""
    if since:
        time_filter = f" AND {_time_expr()} >= :since"
        params["since"] = since

    base_select = [
        "a.id", "a.severity", "a.label", "a.device_id", "a.user_id",
        "a.lat", "a.lon", "a.address",
        "a.emotion", "a.meta",
        "a.raw_path", "a.face_path", "a.image_path", "a.image_face_path",
        f"{_time_expr()} AS ts"
    ]

    where = ""
    join = ""
    if user.role == UserRole.admin:
        where = "" if include_hidden else "WHERE COALESCE(a.hidden,false)=false"
    else:
        join = "LEFT JOIN devices d ON d.id = a.device_id LEFT JOIN alert_access aa ON aa.alert_id = a.id AND aa.user_id = :uid"
        where = "WHERE COALESCE(a.hidden,false)=false AND (d.owner_user_id = :uid OR aa.user_id IS NOT NULL)"

    sql = f"""
        SELECT {', '.join(base_select)},
               COALESCE(a.raw_path, a.image_path) AS raw_rel,
               COALESCE(a.face_path, a.image_face_path) AS face_rel,
               aa.can_view_face AS can_view_face
        FROM alerts a
        {join}
        {where}
        {time_filter}
        ORDER BY {_time_expr()} DESC
        LIMIT :limit
    """

    rows = db.execute(text(sql), params).mappings().all()

    out = []
    for r in rows:
        raw_rel = r.get("raw_rel")
        face_rel = r.get("face_rel")
        # media URLs always under protected /media/alerts
        raw_url = f"/media/alerts/{raw_rel}" if raw_rel else None
        face_url = f"/media/alerts/{face_rel}" if face_rel else None

        # Hide media for citizens by default
        if user.role == UserRole.citizen:
            raw_url = None

        can_face = True
        if user.role == UserRole.citizen:
            can_face = bool(r.get("can_view_face"))
        elif user.role == UserRole.officer:
            if r.get("can_view_face") is False:
                can_face = False

        out.append({
            "id": str(r["id"]),
            "severity": r["severity"],
            "label": r["label"],
            "device_id": str(r["device_id"]) if r["device_id"] else None,
            "user_id": str(r["user_id"]) if r["user_id"] else None,
            "lat": r["lat"],
            "lon": r["lon"],
            "address": r["address"],
            "emotion": r["emotion"],
            "meta": r["meta"] or {},
            "ts_utc": r["ts"].isoformat() if r["ts"] else None,
            "raw_url": raw_url,
            "face_url": face_url if (user.role != UserRole.citizen and can_face) or (user.role == UserRole.citizen and can_face) else None,
        })
    return out

@router.post("/{alert_id}/share")
def share_alert(alert_id: str, payload: ShareAlertRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    require_role(user, {UserRole.admin})
    sql = text("""
        INSERT INTO alert_access (alert_id, user_id, can_view_face)
        VALUES (:aid, :uid, :face)
        ON CONFLICT (alert_id, user_id) DO UPDATE SET can_view_face = EXCLUDED.can_view_face
    """)
    db.execute(sql, {"aid": alert_id, "uid": payload.user_id, "face": bool(payload.can_view_face)})
    db.commit()
    return {"ok": True}

@router.post("/{alert_id}/hide")
def hide_alert(alert_id: str, payload: HideAlertRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    require_role(user, {UserRole.admin})
    db.execute(text("""UPDATE alerts SET hidden = :hidden WHERE id = :aid"""), {"hidden": bool(payload.hidden), "aid": alert_id})
    if payload.reason:
        db.execute(text("""UPDATE alerts SET meta = COALESCE(meta, '{}'::jsonb) || jsonb_build_object('hidden_reason', :r) WHERE id = :aid"""), {"r": payload.reason, "aid": alert_id})
    db.commit()
    return {"ok": True}

@router.get("/map")
def alerts_map(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
    since: Optional[str] = Query(None),
    precision: int = Query(3, ge=1, le=4, description="Rounding precision for lat/lon bins"),
    limit: int = Query(1000, ge=10, le=10000)
):
    require_role(user, {UserRole.admin})
    params = {"limit": limit}
    since_sql = ""
    if since:
        since_sql = f" AND {_time_expr()} >= :since"
        params["since"] = since
    sql = text(f"""
        SELECT ROUND(a.lat::numeric, :prec) AS lat_bin,
               ROUND(a.lon::numeric, :prec) AS lon_bin,
               AVG(a.severity) AS severity_avg,
               COUNT(*) AS cnt,
               MAX({_time_expr()}) AS last_ts
        FROM alerts a
        WHERE a.lat IS NOT NULL AND a.lon IS NOT NULL
          AND COALESCE(a.hidden,false)=false
          {since_sql}
        GROUP BY 1,2
        ORDER BY last_ts DESC
        LIMIT :limit
    """.replace(":prec", str(precision)))
    rows = db.execute(sql, params).mappings().all()
    return [{
        "lat": float(r["lat_bin"]),
        "lon": float(r["lon_bin"]),
        "severity_avg": float(r["severity_avg"]) if r["severity_avg"] is not None else 0.0,
        "count": int(r["cnt"]),
        "last_ts": r["last_ts"].isoformat() if r["last_ts"] else None,
    } for r in rows]
