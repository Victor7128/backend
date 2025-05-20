from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from login import router as auth_router
from filtros.filtro_yape import router as yape_router
from filtros.filtro_fuente import router as fuente_router

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
app.include_router(fuente_router)
