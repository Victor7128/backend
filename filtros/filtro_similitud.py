"""
Filtro SSIM ultra-rápido usando datos pre-procesados
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import io
import asyncio
from typing import Dict, List
import gc

# Importar las constantes generadas
try:
    from image_constants import IMAGE_DATA, IMAGE_ARRAYS, UNIQUE_IMAGE_NAMES, get_image_array  # type: ignore
    CONSTANTS_AVAILABLE = True
    print(f"✅ Constantes cargadas: {len(UNIQUE_IMAGE_NAMES)} imágenes únicas")
except ImportError:
    CONSTANTS_AVAILABLE = False
    UNIQUE_IMAGE_NAMES = []
    IMAGE_DATA = {}
    # Define a dummy get_image_array to avoid unbound errors
    from typing import Optional
    def get_image_array(nombre: str) -> Optional[np.ndarray]:
        # Return a dummy array of zeros with the expected shape
        return np.zeros((128, 128), dtype=np.uint8)
    print("❌ Ejecuta primero: python generate_image_fingerprints.py")

router = APIRouter()

RESIZE_DIM = (128, 128)

def quick_similarity_check(features1: dict, features2: dict) -> float:
    """
    Comparación ultra-rápida usando características pre-calculadas
    Retorna un score aproximado sin SSIM completo
    """
    # Diferencia de medias
    mean_diff = abs(features1["mean"] - features2["mean"]) / 255.0
    
    # Diferencia de desviaciones estándar
    std_diff = abs(features1["std"] - features2["std"]) / 255.0
    
    # Comparación de histogramas (correlación)
    hist1 = np.array(features1["histogram"])
    hist2 = np.array(features2["histogram"])
    hist_corr = np.corrcoef(hist1, hist2)[0, 1]
    hist_corr = 0 if np.isnan(hist_corr) else hist_corr
    
    # Diferencia de gradientes
    grad_diff = abs(features1["gradient_x"] - features2["gradient_x"]) + \
                abs(features1["gradient_y"] - features2["gradient_y"])
    grad_diff = grad_diff / 100.0  # Normalizar
    
    # Score combinado (ponderado)
    similarity = (
        (1 - mean_diff) * 0.2 +
        (1 - std_diff) * 0.2 +
        hist_corr * 0.4 +
        (1 - min(grad_diff, 1.0)) * 0.2
    )
    
    return max(0, min(1, similarity))

def extract_input_features(img_array: np.ndarray) -> dict:
    """Extrae características de la imagen de entrada"""
    mean_val = float(np.mean(img_array))
    std_val = float(np.std(img_array))
    
    hist, _ = np.histogram(img_array.flatten(), bins=16, range=(0, 256))
    hist_normalized = (hist / hist.sum()).tolist() if hist.sum() > 0 else [0] * 16
    
    grad_x = np.abs(np.diff(img_array, axis=1)).mean() if img_array.shape[1] > 1 else 0
    grad_y = np.abs(np.diff(img_array, axis=0)).mean() if img_array.shape[0] > 1 else 0
    
    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "histogram": [round(x, 4) for x in hist_normalized],
        "gradient_x": round(float(grad_x), 2),
        "gradient_y": round(float(grad_y), 2)
    }

def ultra_fast_comparison(img_bytes: bytes, use_full_ssim: bool = False) -> dict:
    """
    Comparación ultra-rápida usando características pre-calculadas
    """
    if not CONSTANTS_AVAILABLE:
        raise ValueError("Constantes no disponibles. Ejecuta generate_image_fingerprints.py")
    
    try:
        # Procesar imagen de entrada
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize(RESIZE_DIM, Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        # Extraer características de entrada
        input_features = extract_input_features(img_array)
        
        resultados = []
        
        # Comparación rápida con características
        for nombre in UNIQUE_IMAGE_NAMES:
            ref_features = IMAGE_DATA[nombre]
            
            # Comparación ultra-rápida
            quick_score = quick_similarity_check(input_features, ref_features)
            quick_pct = round(quick_score * 100, 2)
            
            # Si queremos precisión total, usar SSIM en candidatos prometedores
            if use_full_ssim and quick_pct > 50:  # Solo SSIM para candidatos buenos
                ref_array = get_image_array(nombre)
                if ref_array is not None:
                    ssim_score = ssim(img_array, ref_array, data_range=255)
                    if isinstance(ssim_score, tuple):
                        ssim_score = ssim_score[0]
                    final_pct = round(float(ssim_score) * 100, 2)
                else:
                    final_pct = quick_pct
            else:
                final_pct = quick_pct
            
            resultados.append({
                "imagen": nombre,
                "similitud": final_pct,
                "method": "ssim" if (use_full_ssim and quick_pct > 50) else "quick"
            })
        
        # Calcular promedio
        if resultados:
            similitudes = [r["similitud"] for r in resultados]
            promedio = round(sum(similitudes) / len(similitudes), 2)
        else:
            promedio = 0.0
        
        return {"detalle": resultados, "promedio": promedio}
    
    except Exception as e:
        print(f"Error en comparación: {e}")
        return {"detalle": [], "promedio": 0.0}

@router.post("/filtro_ssim")
async def filtro_ssim(
    file: UploadFile = File(...),
    precision: str = "fast"  # "fast" o "precise"
):
    """
    Filtro SSIM ultra-rápido
    - precision="fast": Solo características (sub-segundo)
    - precision="precise": SSIM completo para candidatos prometedores
    """
    try:
        if not CONSTANTS_AVAILABLE:
            raise HTTPException(
                status_code=500, 
                detail="❌ Sistema no inicializado. Contacta al administrador."
            )
        
        # Validaciones
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=422, detail="❌ Debe ser imagen válida")

        if file.size and file.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="❌ Imagen muy grande (máx 5MB)")
        
        contenido = await file.read()
        if not contenido:
            raise HTTPException(status_code=422, detail="❌ Archivo vacío")

        # Procesamiento ultra-rápido
        use_full_ssim = (precision == "precise")
        
        loop = asyncio.get_event_loop()
        resultado = await asyncio.wait_for(
            loop.run_in_executor(None, ultra_fast_comparison, contenido, use_full_ssim),
            timeout=15.0  # Mucho más rápido
        )
        
        # Estadísticas
        imagenes_alta_coincidencia = sum(
            1 for img in resultado["detalle"] 
            if img["similitud"] > 60
        )
        
        # Obtener top 5 matches
        top_matches = sorted(
            resultado["detalle"], 
            key=lambda x: x["similitud"], 
            reverse=True
        )[:5]
        
        gc.collect()
        
        return {
            "imagenes_comparadas": len(resultado["detalle"]),
            "imagenes_unicas": len(UNIQUE_IMAGE_NAMES),
            "promedio_similitud": resultado['promedio'],
            "imagenes_con_alta_coincidencia": imagenes_alta_coincidencia,
            "top_matches": top_matches,
            "precision_mode": precision,
            "total_time_estimate": "< 1 segundo" if not use_full_ssim else "< 5 segundos"
        }
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="❌ Timeout - intenta modo 'fast'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)[:100]}")