import requests
import os
import cv2
import numpy as np
import re
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

API_KEY = 'K84680463488957'
OCR_API_URL = 'https://api.ocr.space/parse/image'

def enviar_imagen_ocr(path_imagen):
    with open(path_imagen, 'rb') as f:
        response = requests.post(
            OCR_API_URL,
            files={'file': f},
            data={
                'apikey': API_KEY,
                'language': 'spa',
                'OCREngine': '2',
                'isOverlayRequired': True
            }
        )
    resultado = response.json()
    if resultado.get("IsErroredOnProcessing"):
        return None, []
    parsed_result = resultado['ParsedResults'][0]
    return parsed_result['ParsedText'].strip(), parsed_result.get('TextOverlay', {}).get('Lines', [])

def recortar_cuadro_blanco(imagen_path):
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 240, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    contorno_mayor = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)
    recorte = imagen[y:y + h, x:x + w]
    temp_path = "recorte_temporal.png"
    cv2.imwrite(temp_path, recorte)
    return temp_path

def extraer_datos_con_contexto(texto):
    match_monto = re.search(r'S/\s?(\d+(?:\.\d{2})?)', texto)
    if not match_monto:
        return None, "Monto no detectado."
    try:
        monto = float(match_monto.group(1))
        if monto > 500:
            return None, "Monto fuera de rango (mayor a 500)."
    except Exception:
        return None, "Monto inválido."
    receptor = None
    lineas = texto.splitlines()
    linea_monto_idx = -1
    for i, linea in enumerate(lineas):
        if re.search(r'S/\s?(\d+(?:\.\d{2})?)', linea):
            linea_monto_idx = i
            break
    if linea_monto_idx != -1 and linea_monto_idx + 1 < len(lineas):
        posible_nombre = lineas[linea_monto_idx + 1].strip()
        if re.match(r"^[A-Za-zÁÉÍÓÚÑáéíóúñ. ]{3,}$", posible_nombre) and len(posible_nombre.split()) >= 2:
            receptor = posible_nombre
    if not receptor:
        match_destino = re.search(r'Destino[:\s]*([A-Za-zÁÉÍÓÚÑáéíóúñ ]+)', texto)
        if match_destino:
            receptor = match_destino.group(1).strip()
    match_fecha = re.search(
        r'(\d{1,2})\s+([a-zA-ZáéíóúñÑ]{3,9})\.?\s+(\d{4}).*?(\d{1,2}:\d{2}.*?m)',
        texto
    )
    fecha_hora_valida = None
    if match_fecha:
        dia, mes_str, anio, hora = match_fecha.groups()
        meses = {
            'ene': 1, 'enero': 1,'feb': 2, 'febrero': 2,'mar': 3, 'marzo': 3,
            'abr': 4, 'abril': 4,'may': 5, 'mayo': 5,'jun': 6, 'junio': 6,
            'jul': 7, 'julio': 7,'ago': 8, 'agosto': 8,'sep': 9, 'sept': 9, 'septiembre': 9,
            'oct': 10, 'octubre': 10,'nov': 11, 'noviembre': 11,'dic': 12, 'diciembre': 12
        }
        mes_limpio = mes_str.lower().strip('.')
        mes_num = meses.get(mes_limpio, 0)
        try:
            fecha_obj = datetime(int(anio), mes_num, int(dia))
            fecha_formateada = fecha_obj.strftime("%Y-%m-%d")
            fecha_hora_valida = f"{fecha_formateada} {hora}"
        except Exception:
            fecha_hora_valida = None
    nro_operacion = None
    for i, linea in enumerate(lineas):
        if 'Nro. de operación' in linea:
            for extra_i in range(1, 3):
                if i + extra_i < len(lineas):
                    nums = re.findall(r'\d{8,12}', lineas[i + extra_i])
                    if nums:
                        nro_operacion = nums[0]
                        break
            break
    if not nro_operacion:
        posibles = re.findall(r'\b\d{8,12}\b', texto)
        if posibles:
            nro_operacion = posibles[-1]
    return {
        "nro_operacion": nro_operacion,
        "fecha_hora": fecha_hora_valida,
        "receptor": receptor
    }, None

@router.post("/filtro_ocr")
async def filtro_ocr(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=422, detail="❌ El archivo debe ser una imagen")
        content = await file.read()
        temp_path = "temp_ocr_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(content)
        recorte_path = recortar_cuadro_blanco(temp_path)
        if not recorte_path:
            os.remove(temp_path)
            raise HTTPException(status_code=422, detail="❌ No se pudo procesar la imagen.")
        texto, _ = enviar_imagen_ocr(recorte_path)
        os.remove(temp_path)
        os.remove(recorte_path)
        if not texto:
            raise HTTPException(status_code=422, detail="❌ No se pudo extraer texto de la imagen.")
        datos, error = extraer_datos_con_contexto(texto)
        if error:
            raise HTTPException(status_code=422, detail=error)
        if not datos:
            raise HTTPException(status_code=422, detail="Comprobante no válido.")
        return datos
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error en el servidor: {str(e)}")