import cv2
import numpy as np
from typing import Dict, Any, List
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def analizar_gradiente_yape(image: np.ndarray) -> Dict[str, Any]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    purple_region = cv2.bitwise_and(image, image, mask=mask_purple)
    purple_gray = cv2.cvtColor(purple_region, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(purple_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(purple_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    purple_pixels = purple_gray > 0
    if np.sum(purple_pixels) > 0:
        gradient_in_purple = gradient_magnitude[purple_pixels]
        gradient_mean = float(np.mean(gradient_in_purple))
        gradient_std = float(np.std(gradient_in_purple))
        gradient_max = float(np.max(gradient_in_purple))
    else:
        gradient_mean = gradient_std = gradient_max = 0.0
    gradiente_sospechoso = bool(
        gradient_std < 2 or
        gradient_std > 15 or
        gradient_mean > 20
    )
    return {
        'area_morada_pixeles': int(np.sum(purple_pixels)),
        'gradiente_promedio': round(gradient_mean, 2),
        'gradiente_desviacion': round(gradient_std, 2),
        'gradiente_maximo': round(gradient_max, 2),
        'gradiente_sospechoso': gradiente_sospechoso,
        'porcentaje_area_morada': round(float(np.sum(purple_pixels) / image.size * 100), 2)
    }

def analizar_bordes_tarjeta_blanca(image: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "No se encontró tarjeta blanca", "sospechoso": True}
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    is_rectangular = bool(len(approx) == 4)
    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, True))
    if perimeter > 0:
        roundness = float((4 * np.pi * area) / (perimeter * perimeter))
    else:
        roundness = 0.0
    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    bordes_sospechosos = bool(
        not is_rectangular or
        solidity < 0.95 or
        roundness > 0.85
    )
    return {
        'es_rectangular': is_rectangular,
        'vertices_detectados': int(len(approx)),
        'area_tarjeta': int(area),
        'perimetro': round(perimeter, 2),
        'roundness': round(roundness, 3),
        'solidity': round(solidity, 3),
        'bordes_sospechosos': bordes_sospechosos
    }

def analizar_consistencia_texto(image: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((2, 6), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        if (20 < area < 5000 and
            0.2 < aspect_ratio < 8 and
            h > 8):
            text_contours.append((int(x), int(y), int(w), int(h)))
    if len(text_contours) >= 3:
        y_positions = [y for x, y, w, h in text_contours]
        x_positions = [x for x, y, w, h in text_contours]
        y_std = float(np.std(y_positions))
        x_std = float(np.std(x_positions))
        texto_desalineado = bool(y_std > 50 or x_std > 100)
    else:
        y_std = x_std = 0.0
        texto_desalineado = True
    return {
        'regiones_texto_detectadas': int(len(text_contours)),
        'desviacion_y': round(y_std, 2),
        'desviacion_x': round(x_std, 2),
        'texto_desalineado': texto_desalineado
    }

@router.post("/filtro_ruido")
async def filtro_ruido(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=422, detail="❌ El archivo debe ser una imagen")
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=422, detail="❌ No se pudo decodificar la imagen")
        gradiente = analizar_gradiente_yape(image)
        bordes = analizar_bordes_tarjeta_blanca(image)
        texto = analizar_consistencia_texto(image)
        porcentaje = 5 
        if gradiente['gradiente_sospechoso']:
            porcentaje += 25
        if gradiente['porcentaje_area_morada'] < 30:
            porcentaje += 20
        if bordes.get('bordes_sospechosos', True):
            porcentaje += 30
        if bordes.get('solidity', 0) < 0.90:
            porcentaje += 15
        if texto['texto_desalineado']:
            porcentaje += 20
        if texto['regiones_texto_detectadas'] < 5:
            porcentaje += 15
        resultado = {
            "porcentaje_falsificacion": min(int(porcentaje), 95),
            "analisis_gradiente": convert_numpy_types(gradiente),
            "analisis_bordes": convert_numpy_types(bordes),
            "analisis_texto": convert_numpy_types(texto)
        }
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando imagen Yape: {e}")
