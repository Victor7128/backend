#Filtro fuente
import cv2
import numpy as np
from typing import Dict, Tuple

def validate_yape_font(image_bytes: bytes) -> Tuple[float, str]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None or img.size == 0:
        raise ValueError("Imagen corrupta o formato no soportado.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_text_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = float(w) / h if h > 0 else 0
        if area > 100 and aspect_ratio < 10:
            valid_text_contours.append((x, y, w, h))
    
    if len(valid_text_contours) < 10:
        return 0.0, "Veracidad de Fuente"
    
    total_text_area = sum(w * h for _, _, w, h in valid_text_contours)
    image_area = img.shape[0] * img.shape[1]
    text_area_ratio = total_text_area / image_area
    
    dark_pixels = 0
    text_pixels = 0
    for x, y, w, h in valid_text_contours:
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            dark_count = np.sum(roi < 80)
            dark_pixels += dark_count
            text_pixels += roi.size
    
    dark_ratio = dark_pixels / text_pixels if text_pixels > 0 else 0
    
    porcentaje = min(100, (text_area_ratio * 50 + dark_ratio * 50))
    nombre = "Veracidad de Fuente"
    
    return porcentaje, nombre