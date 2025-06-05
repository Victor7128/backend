#Filtro de similitud de imágenes usando SSIM
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import io
import os

router = APIRouter()

REAL_IMAGES_DIR = "./filtros/biblioteca"

def porcentaje_SSIM(img_bytes: bytes) -> dict:
    try:
        img_in = Image.open(io.BytesIO(img_bytes)).convert("L").resize((300, 300))
        arr_in = np.array(img_in)
        resultados = []
        for i in range(1, 75):
            nombre = f"{i}.jpeg"
            path_real = os.path.join(REAL_IMAGES_DIR, nombre)
            if not os.path.exists(path_real):
                continue
            try:
                img_real = Image.open(path_real).convert("L").resize((300, 300))
                arr_real = np.array(img_real)
                result = ssim(arr_in, arr_real, full=True)
                score = result[0]
                pct = round(float(score * 100), 2)
                resultados.append({"imagen": nombre, "similitud": pct})
            except Exception:
                continue
        if not resultados:
            return {"detalle": [], "promedio": 0.0}
        todas = [r["similitud"] for r in resultados]
        promedio = round(float(np.mean(todas)), 2)
        return {"detalle": resultados, "promedio": promedio}
    except Exception:
        return {"detalle": [], "promedio": 0.0}

@router.post("/filtro_ssim")
async def filtro_ssim(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=422, detail="❌ El archivo debe ser una imagen válida")

        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="❌ El archivo es demasiado grande (máximo 10 MB)")
        contenido = await file.read()
        if not contenido:
            raise HTTPException(status_code=422, detail="❌ El archivo está vacío")

        resultado_ssim = porcentaje_SSIM(contenido)
        
        # Contar imágenes con más del 60% de similitud
        imagenes_alta_coincidencia = 0
        if resultado_ssim["detalle"]:
            imagenes_alta_coincidencia = len([
                img for img in resultado_ssim["detalle"] 
                if img["similitud"] > 60
            ])
        
        return {
            "imagenes_comparadas": len(resultado_ssim["detalle"]),
            "promedio_similitud": resultado_ssim['promedio'],
            "imagenes_con_alta_coincidencia": imagenes_alta_coincidencia
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error interno: {e}")
