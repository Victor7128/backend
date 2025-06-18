import requests
from typing import Dict, List, Tuple
import math
from fastapi import APIRouter, File, UploadFile, HTTPException
import tempfile
import os

router = APIRouter()

plantilla1 = [
    {"WordText": "i", "Left": 74, "Top": 321, "Height": 32, "Width": 8},
    {"WordText": "Yapeaste!", "Left": 85, "Top": 313, "Height": 42, "Width": 196},
    {"WordText": "S/", "Left": 73, "Top": 380, "Height": 52, "Width": 63},
    {"WordText": "CÖDIGO", "Left": 73, "Top": 639, "Height": 21, "Width": 83},
    {"WordText": "DE", "Left": 165, "Top": 643, "Height": 17, "Width": 26},
    {"WordText": "SEGURIDAD", "Left": 199, "Top": 643, "Height": 17, "Width": 125},
    {"WordText": "DATOS", "Left": 73, "Top": 757, "Height": 17, "Width": 71},
    {"WordText": "DE", "Left": 153, "Top": 757, "Height": 17, "Width": 26},
    {"WordText": "LA", "Left": 188, "Top": 757, "Height": 17, "Width": 28},
    {"WordText": "TRANSACCIÖN", "Left": 222, "Top": 753, "Height": 21, "Width": 159},
    {"WordText": "Nro.", "Left": 74, "Top": 822, "Height": 23, "Width": 55},
    {"WordText": "de", "Left": 141, "Top": 821, "Height": 24, "Width": 32},
    {"WordText": "celular", "Left": 184, "Top": 821, "Height": 24, "Width": 93},
    {"WordText": "Destino", "Left": 73, "Top": 876, "Height": 24, "Width": 104},
    {"WordText": "Nro.", "Left": 74, "Top": 932, "Height": 23, "Width": 55},
    {"WordText": "de", "Left": 141, "Top": 931, "Height": 24, "Width": 32},
    {"WordText": "operaciön", "Left": 184, "Top": 930, "Height": 31, "Width": 136},
    {"WordText": "Yape", "Left": 697, "Top": 877, "Height": 29, "Width": 70}
]

plantilla2 = [
    {"WordText": "iTe", "Left": 103, "Top": 426, "Height": 55, "Width": 79},
    {"WordText": "Yapearon!", "Left": 199, "Top": 426, "Height": 55, "Width": 286},
    {"WordText": "Sl", "Left": 101, "Top": 542, "Height": 77, "Width": 79},
    {"WordText": "CÖDIGO", "Left": 100, "Top": 896, "Height": 33, "Width": 137},
    {"WordText": "DE", "Left": 250, "Top": 904, "Height": 25, "Width": 42},
    {"WordText": "SEGURIDAD", "Left": 305, "Top": 904, "Height": 25, "Width": 202},
    {"WordText": "DATOS", "Left": 101, "Top": 1061, "Height": 25, "Width": 113},
    {"WordText": "DE", "Left": 228, "Top": 1061, "Height": 24, "Width": 42},
    {"WordText": "LA", "Left": 283, "Top": 1061, "Height": 25, "Width": 42},
    {"WordText": "TRANSACCIÖN", "Left": 336, "Top": 1053, "Height": 33, "Width": 255},
    {"WordText": "Nro.", "Left": 102, "Top": 1157, "Height": 35, "Width": 87},
    {"WordText": "de", "Left": 207, "Top": 1155, "Height": 37, "Width": 52},
    {"WordText": "celular", "Left": 276, "Top": 1155, "Height": 37, "Width": 146},
    {"WordText": "Destino", "Left": 102, "Top": 1244, "Height": 37, "Width": 164},
    {"WordText": "Nro.", "Left": 102, "Top": 1335, "Height": 36, "Width": 87},
    {"WordText": "de", "Left": 207, "Top": 1333, "Height": 38, "Width": 52},
    {"WordText": "operaci6n", "Left": 276, "Top": 1332, "Height": 47, "Width": 218},
    {"WordText": "Yape", "Left": 1081, "Top": 1246, "Height": 44, "Width": 109}
]

OCR_API_KEY = "K84911188188957"

async def ocr_api_async(file_path: str, api_key: str) -> Dict:
    url = "https://api.ocr.space/parse/image"
    
    payload = {
        'isTable': 'true',
        'apikey': api_key,
        'language': 'spa',
        'isOverlayRequired': 'true'
    }
    
    try:
        with open(file_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(url, data=payload, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            return {}
            
    except Exception as e:
        return {}

def normalizar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.lower().strip()
    reemplazos = {
        'ö': 'o', 'ü': 'u', 'ä': 'a', 'ñ': 'n',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'código': 'codigo', 'transacción': 'transaccion',
        'operación': 'operacion'
    }
    for original, reemplazo in reemplazos.items():
        texto = texto.replace(original, reemplazo)
    return texto

def calcular_distancia(pos1: Dict, pos2: Dict) -> float:
    x1, y1 = pos1.get('Left', 0), pos1.get('Top', 0)
    x2, y2 = pos2.get('Left', 0), pos2.get('Top', 0)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def extraer_palabras_ocr(data_ocr: Dict) -> List[Dict]:
    palabras = []
    try:
        if 'ParsedResults' in data_ocr and data_ocr['ParsedResults']:
            for resultado in data_ocr['ParsedResults']:
                if 'TextOverlay' in resultado and 'Lines' in resultado['TextOverlay']:
                    for linea in resultado['TextOverlay']['Lines']:
                        if 'Words' in linea:
                            for palabra in linea['Words']:
                                palabras.append({
                                    'WordText': palabra.get('WordText', ''),
                                    'Left': palabra.get('Left', 0),
                                    'Top': palabra.get('Top', 0),
                                    'Height': palabra.get('Height', 0),
                                    'Width': palabra.get('Width', 0)
                                })
    except Exception as e:
        print(f"Error extrayendo palabras OCR: {str(e)}")
    
    return palabras

def comparar_plantilla(plantilla: List[Dict], palabras_ocr: List[Dict], nombre: str) -> Tuple[float, Dict]:
    if not plantilla or not palabras_ocr:
        return 0.0, {}
    
    coincidencias = 0
    coincidencias_exactas = 0
    coincidencias_parciales = 0
    detalles = []
    palabras_usadas = set()
    
    for palabra_plantilla in plantilla:
        texto_plantilla = normalizar_texto(palabra_plantilla.get('WordText', ''))
        mejor_coincidencia = None
        mejor_puntuacion = 0.0
        mejor_indice = -1
        for i, palabra_ocr in enumerate(palabras_ocr):
            if i in palabras_usadas:
                continue
                
            texto_ocr = normalizar_texto(palabra_ocr.get('WordText', ''))
            if texto_plantilla == texto_ocr:
                similitud_texto = 1.0
            elif texto_plantilla in texto_ocr or texto_ocr in texto_plantilla:
                if len(texto_plantilla) >= 3 and len(texto_ocr) >= 3:
                    similitud_texto = 0.8
                else:
                    similitud_texto = 0.6
            else:
                if len(texto_plantilla) > 0 and len(texto_ocr) > 0:
                    comunes = set(texto_plantilla) & set(texto_ocr)
                    similitud_texto = len(comunes) / max(len(set(texto_plantilla)), len(set(texto_ocr)))
                else:
                    similitud_texto = 0.0
            
            if similitud_texto > 0.3:
                distancia = calcular_distancia(palabra_plantilla, palabra_ocr)
                
                umbral_distancia = 200
                if distancia <= umbral_distancia:
                    puntuacion_posicion = (umbral_distancia - distancia) / umbral_distancia
                else:
                    puntuacion_posicion = 0.0
                
                puntuacion_total = (similitud_texto * 0.7) + (puntuacion_posicion * 0.3)
                
                if puntuacion_total > mejor_puntuacion:
                    mejor_puntuacion = puntuacion_total
                    mejor_coincidencia = palabra_ocr
                    mejor_indice = i
        
        if mejor_coincidencia and mejor_puntuacion > 0.5:
            coincidencias += 1
            palabras_usadas.add(mejor_indice)
            
            if mejor_puntuacion >= 0.9:
                coincidencias_exactas += 1
                estado = "EXACTA"
            else:
                coincidencias_parciales += 1
                estado = "PARCIAL"
            
            distancia = calcular_distancia(palabra_plantilla, mejor_coincidencia)
        else:
            print(f"✗ '{palabra_plantilla.get('WordText', '')}' -> NO ENCONTRADA")

        detalles.append({
            'palabra_plantilla': palabra_plantilla.get('WordText', ''),
            'coincidencia': mejor_coincidencia.get('WordText', '') if mejor_coincidencia else None,
            'puntuacion': mejor_puntuacion,
            'distancia': calcular_distancia(palabra_plantilla, mejor_coincidencia) if mejor_coincidencia else float('inf')
        })
    porcentaje_coincidencias = (coincidencias / len(plantilla)) * 100
    
    resultado = {
        'coincidencias_totales': coincidencias,
        'coincidencias_exactas': coincidencias_exactas,
        'coincidencias_parciales': coincidencias_parciales,
        'total_palabras': len(plantilla),
        'porcentaje': porcentaje_coincidencias,
        'detalles': detalles
    }    
    return porcentaje_coincidencias, resultado

def comparar_plantillas(plantilla1: List[Dict], plantilla2: List[Dict], data_ocr: Dict) -> Dict:
    palabras_ocr = extraer_palabras_ocr(data_ocr)
    
    if not palabras_ocr:
        print("No se pudieron extraer palabras del OCR")
        return {'error': 'No se pudieron extraer palabras del OCR', 'porcentaje': 0}
    porcentaje1, detalles1 = comparar_plantilla(plantilla1, palabras_ocr, "Plantilla 1")
    porcentaje2, detalles2 = comparar_plantilla(plantilla2, palabras_ocr, "Plantilla 2")
    
    if porcentaje1 > porcentaje2:
        ganadora = "Plantilla 1"
        mejor_porcentaje = porcentaje1
        mejor_detalles = detalles1
    elif porcentaje2 > porcentaje1:
        ganadora = "Plantilla 2"
        mejor_porcentaje = porcentaje2
        mejor_detalles = detalles2
    else:
        ganadora = "Empate"
        mejor_porcentaje = porcentaje1
        mejor_detalles = detalles1
    
    return {
        'porcentaje': round(mejor_porcentaje, 2),
    }

@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    temp_file = None
    temp_file_path = None
    try:
        file_content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        data_ocr = await ocr_api_async(temp_file_path, OCR_API_KEY)
        
        if not data_ocr:
            raise HTTPException(status_code=500, detail="Error procesando la imagen con OCR")
        resultado = comparar_plantillas(plantilla1, plantilla2, data_ocr)        
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error procesando la imagen: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"No se pudo eliminar archivo temporal: {e}")