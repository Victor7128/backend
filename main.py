from fastapi import FastAPI
from login import router as auth_router

app = FastAPI()
app.include_router(auth_router)