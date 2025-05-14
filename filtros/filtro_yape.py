import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException

router = FastAPI()

class NotYapeTransaction(Exception):
    """Raised when la imagen no es una transacción Yape."""
    pass

def is_yape_transaction(image_bytes: bytes,
                        purple_ratio_thresh: float = 0.4,
                        keyword_ratio_thresh: float = 0.5) -> bool:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("No se pudo decodificar la imagen.")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array((120, 30, 50)),
                       np.array((160, 255, 255)))
    purple_ratio = mask.sum() / (255 * img.size)
    has_purple = purple_ratio > purple_ratio_thresh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresh, lang='spa', config='--psm 6')
    text = text.lower()
    keywords = [
        'yapeaste', 'te yapearon', 's/', 'código de seguridad',
        'datos de la transacción', 'nro. de operación'
    ]
    found = sum(1 for kw in keywords if kw in text)
    kw_ratio = found / len(keywords)
    if (has_purple and kw_ratio >= keyword_ratio_thresh) or kw_ratio >= 0.6:
        return True
    raise NotYapeTransaction("La imagen no es una transacción de Yape.")

@router.post("/filter_yape")
async def filter_yape(file: UploadFile = File(...)):
    img_bytes = await file.read()
    try:
        is_yape_transaction(img_bytes)  
        return {"ok": True}
    except NotYapeTransaction as e:
        raise HTTPException(status_code=400, detail=str(e))  
    except Exception:
        raise HTTPException(status_code=500, detail="Error procesando la imagen")
