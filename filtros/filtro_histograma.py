from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io

router = APIRouter()

@router.post("/histograma")
async def histograma(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen. Asegúrese de que el archivo sea una imagen válida.")

    try:
        histogram = image.histogram()
        r = histogram[0:256]
        g = histogram[256:512]
        b = histogram[512:768]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al calcular el histograma de la imagen.")

    return {
        "r": r,
        "g": g,
        "b": b
    }