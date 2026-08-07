import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import auth_routes, report_routes
from app.firebase_config import UPLOADS_DIR, firebase_initialized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="Intelligent Traffic Safety & Road Infrastructure Reporting System API",
    description="FastAPI Backend for YOLOv8 AI Road Hazard Detection & Firebase Firestore / Storage integration",
    version="1.0.0"
)

# CORS Configuration
origins = [
    "*",  # Allow all origins for seamless development & cross-origin frontend communication
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for local uploaded/annotated road images
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Mount API Routers
app.include_router(auth_routes.router, prefix="/api")
app.include_router(report_routes.router, prefix="/api")


@app.get("/")
def root_status():
    return {
        "system": "Intelligent Traffic Safety & Road Infrastructure Reporting System",
        "status": "online",
        "firebase_mode": "Cloud Firestore & Storage" if firebase_initialized else "Local Storage Engine",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "firebase_connected": firebase_initialized,
        "detector_status": "active"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
