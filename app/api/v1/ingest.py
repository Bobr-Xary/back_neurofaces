import base64, io
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from app.db.session import get_db
from app.core.deps import get_current_device
from app.schemas.alert import AlertIngestRequest, AlertOut
from app.services.alerts_service import create_alert
from app.services.alert_acl import grant_alert_access
from app.models.user import User
from app.models.enums import UserRole
from app.utils.media import save_b64_image

# existing pipeline
from face_recognizer_insightface import FaceRecognizer
from emotion_recognizer import analyze_emotions

router = APIRouter(prefix="/ingest", tags=["ingest"])

_face_recognizer = FaceRecognizer()

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

@router.post("/frame", response_model=AlertOut)
def ingest_frame(payload: AlertIngestRequest, db: Session = Depends(get_db), device=Depends(get_current_device)):
    # 1) decode & save original
    try:
        pil_img = Image.open(io.BytesIO(base64.b64decode(payload.image_b64))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image_b64")
    image_path = save_b64_image(payload.image_b64, "alerts/raw")

    # 2) run face recognition (compat with your server.py style)
    face_results = _face_recognizer.process_image(base64.b64decode(payload.image_b64))  # [] or list of dicts

    face_id = None
    face_img_path = None
    emotion = None
    severity = 0

    if face_results:
        face = face_results[0]
        face_id = face.get("face_id")
        face_img = face.get("face")  # BGR numpy array
        if face_img is not None:
            ok, buf = cv2.imencode(".jpg", face_img)
            if ok:
                fpath = Path("alerts/faces"); fpath.mkdir(parents=True, exist_ok=True)
                fname = fpath / f"{face_id}.jpg"
                with open(fname, "wb") as f:
                    f.write(buf.tobytes())
                face_img_path = str(fname)
            try:
                em = analyze_emotions(face_img)
            except Exception:
                bgr = _pil_to_bgr(pil_img)
                em = analyze_emotions(bgr)
            if isinstance(em, dict):
                emotion = em
                try:
                    severity = int(round(float(em.get("aggression_score", 0))))
                except Exception:
                    severity = 0
            else:
                emotion = {"raw": str(em)}
                severity = 0
    else:
        # fallback: analyze full frame
        try:
            bgr = _pil_to_bgr(pil_img)
            em = analyze_emotions(bgr)
            if isinstance(em, dict):
                emotion = em
                severity = int(round(float(em.get("aggression_score", 0))))
            else:
                emotion = {"raw": str(em)}
        except Exception:
            pass

    lat = None; lon = None
    if payload.location:
        lat = payload.location.get("lat"); lon = payload.location.get("lon")

    alert = create_alert(
        db,
        device_id=str(device.id),
        severity=max(0, min(severity, 10)),
        label="aggression",
        face_id=face_id,
        emotion=emotion,
        lat=lat, lon=lon,
        address=None,
        image_path=image_path,
        image_face_path=face_img_path,
        meta=payload.meta or {},
    )
    owner_id = str(device.owner_user_id) if device.owner_user_id else None
    if owner_id:
        owner_role = db.query(User).filter(User.id == device.owner_user_id).with_entities(User.role).scalar()
        can_face = (owner_role == UserRole.officer)
        grant_alert_access(db, str(alert.id), owner_id, can_view_face=can_face)
    db.commit()
    return alert
