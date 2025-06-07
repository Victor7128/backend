#!/usr/bin/env python3
"""
Script para pre-procesar la biblioteca de imágenes y generar constantes estáticas
Ejecutar una sola vez: python generate_image_fingerprints.py
"""

from PIL import Image
import numpy as np
import os
import json
import hashlib
from collections import defaultdict

REAL_IMAGES_DIR = "./filtros/biblioteca"
RESIZE_DIM = (128, 128)

def get_image_hash(img_array):
    """Genera hash único para detectar imágenes duplicadas"""
    return hashlib.md5(img_array.tobytes()).hexdigest()

def extract_image_features(img_array):
    """Extrae características clave de la imagen para comparación rápida"""
    # Estadísticas básicas
    mean_val = float(np.mean(img_array))
    std_val = float(np.std(img_array))
    
    # Histograma simplificado (16 bins)
    hist, _ = np.histogram(img_array.flatten(), bins=16, range=(0, 256))
    hist_normalized = (hist / hist.sum()).tolist()
    
    # Gradientes (bordes)
    grad_x = np.abs(np.diff(img_array, axis=1)).mean()
    grad_y = np.abs(np.diff(img_array, axis=0)).mean()
    
    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "histogram": [round(x, 4) for x in hist_normalized],
        "gradient_x": round(float(grad_x), 2),
        "gradient_y": round(float(grad_y), 2),
        "array": img_array.tolist()  # Para SSIM exacto si es necesario
    }

def analyze_image_library():
    """Analiza toda la biblioteca y genera datos optimizados"""
    print("🔍 Analizando biblioteca de imágenes...")
    
    image_data = {}
    duplicates = defaultdict(list)
    unique_images = {}
    
    for i in range(1, 75):
        nombre = f"{i}.jpeg"
        path_real = os.path.join(REAL_IMAGES_DIR, nombre)
        
        if not os.path.exists(path_real):
            print(f"❌ No encontrado: {nombre}")
            continue
            
        try:
            with Image.open(path_real) as img:
                # Información original
                original_size = img.size
                file_size = os.path.getsize(path_real)
                
                # Procesar imagen
                img_processed = img.convert("L").resize(RESIZE_DIM, Image.Resampling.LANCZOS)
                img_array = np.array(img_processed)
                
                # Generar hash para detectar duplicados
                img_hash = get_image_hash(img_array)
                
                # Extraer características
                features = extract_image_features(img_array)
                
                image_info = {
                    "nombre": nombre,
                    "original_size": original_size,
                    "file_size_bytes": file_size,
                    "processed_size": RESIZE_DIM,
                    "hash": img_hash,
                    "features": features
                }
                
                # Detectar duplicados
                if img_hash in duplicates:
                    duplicates[img_hash].append(nombre)
                    print(f"🔄 Duplicado detectado: {nombre} (igual a {duplicates[img_hash][0]})")
                else:
                    duplicates[img_hash] = [nombre]
                    unique_images[nombre] = image_info
                
                image_data[nombre] = image_info
                print(f"✅ Procesado: {nombre} ({original_size}) -> {file_size//1024}KB")
                
        except Exception as e:
            print(f"❌ Error procesando {nombre}: {e}")
            continue
    
    # Resumen
    total_images = len(image_data)
    unique_count = len(unique_images)
    duplicate_count = total_images - unique_count
    
    print(f"\n📊 RESUMEN:")
    print(f"   Total imágenes encontradas: {total_images}")
    print(f"   Imágenes únicas: {unique_count}")
    print(f"   Duplicados detectados: {duplicate_count}")
    
    if duplicate_count > 0:
        print(f"\n🔄 DUPLICADOS ENCONTRADOS:")
        for hash_val, names in duplicates.items():
            if len(names) > 1:
                print(f"   {' = '.join(names)}")
    
    # Guardar datos
    output = {
        "metadata": {
            "total_images": total_images,
            "unique_images": unique_count,
            "duplicates": duplicate_count,
            "resize_dim": RESIZE_DIM,
            "generated_at": "2025-06-07 22:27:49"
        },
        "all_images": image_data,
        "unique_images": unique_images,
        "duplicates": dict(duplicates)
    }
    
    # Guardar como JSON
    with open("image_fingerprints.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # Generar constantes Python
    generate_python_constants(unique_images)
    
    print(f"\n✅ Datos guardados en:")
    print(f"   - image_fingerprints.json")
    print(f"   - image_constants.py")
    
    return output

def generate_python_constants(unique_images):
    """Genera archivo Python con constantes estáticas"""
    
    python_code = '''"""
Constantes de imágenes pre-procesadas
Generado automáticamente - NO EDITAR MANUALMENTE
"""
import numpy as np

# Configuración
RESIZE_DIM = (128, 128)
TOTAL_UNIQUE_IMAGES = {total}

# Datos de imágenes únicas (solo estas se comparan)
IMAGE_DATA = {{
{image_data}
}}

# Arrays pre-procesados para SSIM rápido
IMAGE_ARRAYS = {{
{image_arrays}
}}

# Función helper para obtener array de imagen
def get_image_array(nombre: str) -> np.ndarray:
    """Obtiene el array numpy de una imagen por nombre"""
    if nombre in IMAGE_ARRAYS:
        return np.array(IMAGE_ARRAYS[nombre], dtype=np.uint8)
    return None

# Lista de nombres únicos para iteración
UNIQUE_IMAGE_NAMES = {unique_names}
'''.format(
        total=len(unique_images),
        image_data=',\n'.join([
            f'    "{name}": {{\n        "mean": {data["features"]["mean"]},\n        "std": {data["features"]["std"]},\n        "histogram": {data["features"]["histogram"]},\n        "gradient_x": {data["features"]["gradient_x"]},\n        "gradient_y": {data["features"]["gradient_y"]},\n        "original_size": {data["original_size"]},\n        "file_size": {data["file_size_bytes"]}\n    }}'
            for name, data in unique_images.items()
        ]),
        image_arrays=',\n'.join([
            f'    "{name}": {data["features"]["array"]}'
            for name, data in unique_images.items()
        ]),
        unique_names=list(unique_images.keys())
    )
    
    with open("image_constants.py", "w") as f:
        f.write(python_code)

if __name__ == "__main__":
    analyze_image_library()