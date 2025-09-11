from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db
from app.core.deps import get_current_user
from app.schemas.alerts_ext import ShareAlertRequest, HideAlertRequest

router = APIRouter(prefix="/alerts", tags=["alerts-ext"])

def _time_expr():
    return "a.ts_utc"

@router.get("", response_model=List[Dict[str, Any]])
def list_alerts(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=500),
    since: Optional[str] = Query(None, description="ISO datetime; return alerts since this time"),
    include_hidden: bool = Query(False, description="Admin-only: include hidden alerts (ignored for admin — admin sees all)"),
):
    uid = str(user.id)

    # робастная проверка роли
    role_raw = getattr(user, "role", None)
    role_str = getattr(role_raw, "value", role_raw)
    role_str = (str(role_str or "")).lower()
    is_admin   = role_str == "admin"
    is_officer = role_str == "officer"
    is_citizen = role_str in ("citizen", "user")

    params = {"uid": uid, "limit": limit}
    if since:
        params["since"] = since

    # join’ы для владения и ACL
    join_sql = (
        "LEFT JOIN devices d ON d.id = a.device_id "
        "LEFT JOIN alert_access aa ON aa.alert_id = a.id AND aa.user_id = :uid"
    )

    # Базовый SELECT: берём только реально существующие колонки (raw_path/face_path)
    base_select = [
        "a.id", "a.severity", "a.label", "a.device_id", "a.user_id",
        "a.lat", "a.lon", "a.address",
        "a.emotion", "a.meta",
        "a.raw_path", "a.face_path",
        "a.hidden",
        f"{_time_expr()} AS ts",
        "d.owner_user_id AS owner_user_id",
        "(d.owner_user_id = :uid) AS is_owner",
        "(a.user_id = :uid) AS is_creator",      # <-- офицер-инициатор
        "aa.can_view_face AS can_view_face",
    ]

    # WHERE-условия собираем списком
    where_clauses: List[str] = []

    if not is_admin:
        # не-админам скрытые НЕ показываем
        where_clauses.append("COALESCE(a.hidden, false) = false")

        if is_officer:
            # офицер видит: свои девайсы, расшаренные, ИЛИ свои (где он создатель)
            where_clauses.append("(d.owner_user_id = :uid OR aa.user_id IS NOT NULL OR a.user_id = :uid)")
        else:
            # citizen: свои девайсы или расшаренные (как раньше)
            where_clauses.append("(d.owner_user_id = :uid OR aa.user_id IS NOT NULL)")

    # для админа — без ограничений по hidden/владению
    if since:
        where_clauses.append(f"{_time_expr()} >= :since")

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = f"""
        SELECT {', '.join(base_select)}
          FROM alerts a
          {join_sql}
          {where_sql}
         ORDER BY {_time_expr()} DESC
         LIMIT :limit
    """

    rows = db.execute(text(sql), params).mappings().all()

    out: List[Dict[str, Any]] = []
    for r in rows:
        raw_rel = r.get("raw_path")
        face_rel = r.get("face_path")

        # защищённые ссылки через наш /api/v1/media/alerts/*
        raw_url = f"/api/v1/media/alerts/{raw_rel}" if raw_rel else None
        face_url = f"/api/v1/media/alerts/{face_rel}" if face_rel else None

        is_owner    = bool(r.get("is_owner"))
        is_creator  = bool(r.get("is_creator"))
        acl_can_face = r.get("can_view_face")  # None/True/False

        # Права на лицо по ролям
        if is_admin:
            can_face = True
        elif is_officer:
            # офицер видит лицо, если он владелец девайса ИЛИ создатель, иначе по ACL
            can_face = True if (is_owner or is_creator) else bool(acl_can_face)
        elif is_citizen:
            can_face = bool(acl_can_face)
        else:
            can_face = False

        # Гражданским raw никогда не показываем
        if is_citizen:
            raw_url = None
            face_url = face_url if can_face else None
        else:
            face_url = face_url if can_face else None

        # user_id в ответе: если пустой — подставим owner_user_id
        resp_user_id = r.get("user_id") or r.get("owner_user_id")

        out.append({
            "id": str(r["id"]),
            "severity": r["severity"],
            "label": r["label"],
            "device_id": str(r["device_id"]) if r["device_id"] else None,
            "user_id": str(resp_user_id) if resp_user_id else None,
            "lat": r["lat"],
            "lon": r["lon"],
            "address": r["address"],
            "emotion": r["emotion"],
            "meta": r["meta"] or {},
            "ts_utc": r["ts"].isoformat() if r["ts"] else None,
            "raw_url": raw_url,
            "face_url": face_url,
            "hidden": bool(r.get("hidden")),
        })
    return out

@router.post("/{alert_id}/share")
def share_alert(
    alert_id: str,
    payload: ShareAlertRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    role_raw = getattr(user, "role", None)
    role_str = getattr(role_raw, "value", role_raw)
    if str(role_str).lower() != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    sql = text("""
        INSERT INTO alert_access (alert_id, user_id, can_view_face)
        VALUES (:aid, :uid, :face)
        ON CONFLICT (alert_id, user_id) DO UPDATE SET can_view_face = EXCLUDED.can_view_face
    """)
    db.execute(sql, {"aid": alert_id, "uid": payload.user_id, "face": bool(payload.can_view_face)})
    db.commit()
    return {"ok": True}

@router.post("/{alert_id}/hide")
def toggle_hide_alert(
    alert_id: str,
    payload: Optional[HideAlertRequest] = None,
    user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    role_raw = getattr(user, "role", None)
    role_str = getattr(role_raw, "value", role_raw)
    if str(role_str).lower() != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    row = db.execute(
        text("SELECT hidden, meta FROM alerts WHERE id = :aid FOR UPDATE"),
        {"aid": alert_id}
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")

    current_hidden = bool(row.get("hidden"))
    new_hidden = not current_hidden
    reason = (payload.reason.strip() if payload and payload.reason else None)

    # meta-обновление: при скрытии — пишем hidden_reason; при показе — удаляем
    if new_hidden:
        # скрываем
        db.execute(text("""
            UPDATE alerts
               SET hidden = TRUE,
                   meta = COALESCE(meta, '{}'::jsonb) || CASE
                        WHEN :r IS NOT NULL THEN jsonb_build_object('hidden_reason', :r)
                        ELSE '{}'::jsonb
                   END
             WHERE id = :aid
        """), {"aid": alert_id, "r": reason})
    else:
        # делаем видимым + чистим hidden_reason
        db.execute(text("""
            UPDATE alerts
               SET hidden = FALSE,
                   meta = CASE
                            WHEN meta ? 'hidden_reason'
                              THEN meta - 'hidden_reason'
                            ELSE meta
                         END
             WHERE id = :aid
        """), {"aid": alert_id})

    db.commit()
    return {"ok": True, "hidden": new_hidden, "reason": reason if new_hidden else None}

@router.get("/map")
def alerts_map(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
    since: Optional[str] = Query(None),
    precision: int = Query(3, ge=1, le=4, description="Rounding precision for lat/lon bins"),
    limit: int = Query(1000, ge=10, le=10000),
):
    # карта — только для админа
    role_raw = getattr(user, "role", None)
    role_str = getattr(role_raw, "value", role_raw)
    if str(role_str).lower() != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    params = {"limit": limit}
    where_clauses = ["a.lat IS NOT NULL", "a.lon IS NOT NULL"]

    if since:
        params["since"] = since
        where_clauses.append(f"{_time_expr()} >= :since")

    where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = text(f"""
        SELECT ROUND(a.lat::numeric, {precision}) AS lat_bin,
               ROUND(a.lon::numeric, {precision}) AS lon_bin,
               AVG(a.severity) AS severity_avg,
               COUNT(*) AS cnt,
               MAX({_time_expr()}) AS last_ts
          FROM alerts a
          {where_sql}
         GROUP BY 1,2
         ORDER BY last_ts DESC
         LIMIT :limit
    """)
    rows = db.execute(sql, params).mappings().all()
    return [{
        "lat": float(r["lat_bin"]),
        "lon": float(r["lon_bin"]),
        "severity_avg": float(r["severity_avg"]) if r["severity_avg"] is not None else 0.0,
        "count": int(r["cnt"]),
        "last_ts": r["last_ts"].isoformat() if r["last_ts"] else None,
    } for r in rows]
