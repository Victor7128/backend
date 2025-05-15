import cv2
import numpy as np
import pytesseract
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

class NotYapeTransaction(Exception):
    pass

def is_yape_transaction(image_bytes: bytes,
                        purple_ratio_thresh: float = 0.4,
                        keyword_ratio_thresh: float = 0.5) -> bool:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("La imagen está corrupta o tiene un formato no soportado.")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array((120, 30, 50)),
                       np.array((160, 255, 255)))
    purple_ratio = mask.sum() / (255 * img.shape[0] * img.shape[1])
    has_purple = purple_ratio > purple_ratio_thresh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    custom_config = '--psm 6' if np.mean(thresh) > 127 else '--psm 3'
    text = pytesseract.image_to_string(thresh, lang='spa', config=custom_config)
    text = pytesseract.image_to_string(thresh, lang='spa', config=custom_config)
    text = text.lower()
    keywords = [
        'yapeaste', 'te yapearon', 's/', 'código de seguridad']
    found = 0
    for kw in keywords:
        if kw in text:
            found += 1
            if found / len(keywords) >= keyword_ratio_thresh:
                break
    kw_ratio = found / len(keywords)
    if (has_purple and kw_ratio >= keyword_ratio_thresh) or kw_ratio >= 0.6:
        return True
    raise NotYapeTransaction("La imagen no es una transacción de Yape.")


@router.post("/filter_yape")
async def filter_yape(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        is_yape_transaction(img_bytes)  
        return {"ok": True}
    except NotYapeTransaction as e:
        raise HTTPException(status_code=400, detail=str(e))  
    except Exception:
        raise HTTPException(status_code=500, detail="Error procesando la imagen")