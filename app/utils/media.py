import base64, uuid
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_b64_image(image_b64: str, out_dir: str, ext: str = ".jpg") -> str:
    ensure_dir(out_dir)
    raw = base64.b64decode(image_b64)
    fname = f"{uuid.uuid4()}{ext}"
    fpath = Path(out_dir) / fname
    with open(fpath, "wb") as f:
        f.write(raw)
    return str(fpath)
