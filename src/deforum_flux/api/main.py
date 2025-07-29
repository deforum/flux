"""
Main FastAPI application for Deforum Flux
Designed for both local development and RunPod deployment
Simplified version with only existing routes
"""

import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Only import routes that actually exist
from deforum_flux.api.routes.generation import router as generation_router
from deforum_flux.api.routes.models import router as models_router
from deforum.core.logging_config import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Deforum Alpha API")
    yield
    logger.info("Shutting down Deforum Alpha API")

# Initialize FastAPI app
app = FastAPI(
    title="Deforum Alpha API",
    version="1.0.0",
    description="Interactive API for Deforum Flux video generation",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
)

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Deforum Alpha API",
        "version": "1.0.0",
        "description": "Interactive API for Deforum Flux video generation",
        "status": "healthy"
    }

# Include only existing routes
app.include_router(generation_router, prefix="/api/v1", tags=["generation"])
app.include_router(models_router, prefix="/api/v1", tags=["models"])

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "7860"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
