# main
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from login import router as auth_router
from filtros.filtro_yape import router as yape_router
from filtros.filtro_pixeles import router as pixeles_router
from filtros.filtro_exif import router as exif_router
from filtros.filtro_ruido import router as ruido_router
from filtros.filtro_histograma import router as histograma_router
from filtros.filtro_ocr import router as ocr_router
from filtros.filtro_logo import router as logo_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_methods=["*"],       
    allow_headers=["*"],      
    allow_credentials=False,   
)

app.include_router(auth_router)
app.include_router(yape_router)
app.include_router(pixeles_router)
app.include_router(exif_router)
app.include_router(ruido_router)
app.include_router(histograma_router)
app.include_router(ocr_router)
app.include_router(logo_router)