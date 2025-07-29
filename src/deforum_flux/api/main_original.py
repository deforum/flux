"""
Main FastAPI application for Deforum Flux
Designed for both local development and RunPod deployment
Fixed: CORS OPTIONS handling and performance improvements
"""

import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from deforum_flux.api.routes.generation import router as generation_router
from deforum_flux.api.routes.websocket import router as websocket_router
from deforum_flux.api.routes.presets import router as presets_router
from deforum_flux.api.routes.config import router as config_router
from deforum_flux.api.routes.animation import router as animation_router
from deforum_flux.api.routes.payload import router as payload_router
from deforum_flux.api.routes.models import router as models_router
from deforum_flux.api.routes.models_management import router as models_management_router
from deforum.core.logging_config import get_logger

logger = get_logger(__name__)

# Models routes are now split into basic and management modules
# No need for enhanced routes as functionality is consolidated

try:
    from deforum_flux.api.routes.model_health import router as model_health_router
    MODEL_HEALTH_AVAILABLE = True
except ImportError:
    logger.warning("Model health routes not available")
    MODEL_HEALTH_AVAILABLE = False

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

# CORS middleware for browser access - Fixed OPTIONS handling
allowed_origins = []
if os.getenv("RUNPOD_MODE") or os.getenv("PRODUCTION"):
    # Production: restrict to known domains
    allowed_origins = [
        "https://your-domain.com",  # Replace with actual domain
        "https://api.runpod.ai",
    ]
else:
    # Development: allow local development and testing
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:7860", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:7860",
        "*"  # Allow all origins in development for testing
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if os.getenv("RUNPOD_MODE") or os.getenv("PRODUCTION") else ["*"],
    allow_credentials=False,  # Disabled for security - implement proper auth if needed
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
    expose_headers=["Content-Type", "X-Total-Count"]
)

# Add explicit OPTIONS handler for all routes
@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    """Handle all OPTIONS requests properly for CORS preflight"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept, X-Requested-With",
            "Access-Control-Max-Age": "86400"
        }
    )

# Enhanced health check for comprehensive monitoring
@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed system metrics"""
    import torch
    import psutil
    import platform
    from datetime import datetime
    
    try:
        # Basic system info
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        # GPU memory info
        gpu_memory_info = {}
        if gpu_available:
            for i in range(gpu_count):
                mem_info = torch.cuda.get_device_properties(i)
                mem_allocated = torch.cuda.memory_allocated(i)
                mem_cached = torch.cuda.memory_reserved(i)
                gpu_memory_info[f"gpu_{i}"] = {
                    "name": mem_info.name,
                    "total_memory": mem_info.total_memory,
                    "allocated_memory": mem_allocated,
                    "cached_memory": mem_cached,
                    "free_memory": mem_info.total_memory - mem_cached
                }
        
        # System memory
        memory = psutil.virtual_memory()
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Reduced interval for performance
        
        # Disk usage for output directory
        output_dir = Path(__file__).parent.parent / "outputs"
        disk_usage = None
        if output_dir.exists():
            disk_usage = psutil.disk_usage(str(output_dir))
        
        # Environment detection
        environment = "local"
        if os.getenv("RUNPOD_MODE"):
            environment = "runpod"
        elif os.getenv("PRODUCTION"):
            environment = "production"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": environment,
            "system": {
                "platform": platform.system(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count()
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "gpu": {
                "available": gpu_available,
                "count": gpu_count,
                "details": gpu_memory_info
            },
            "disk": {
                "total": disk_usage.total if disk_usage else None,
                "used": disk_usage.used if disk_usage else None,
                "free": disk_usage.free if disk_usage else None,
                "percent": (disk_usage.used / disk_usage.total * 100) if disk_usage else None
            },
            "api": {
                "host": os.getenv("API_HOST", "0.0.0.0"),
                "port": int(os.getenv("API_PORT", "7860")),
                "frontend_served": bool(os.getenv("RUNPOD_MODE") or os.getenv("PRODUCTION"))
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }

# Detailed system status endpoint
@app.get("/api/v1/system/status")
async def system_status():
    """Detailed system status for monitoring and diagnostics"""
    import torch
    import psutil
    from datetime import datetime, timedelta
    
    try:
        # Process information
        process = psutil.Process()
        
        # Uptime calculation
        create_time = datetime.fromtimestamp(process.create_time())
        uptime = datetime.now() - create_time
        
        # GPU utilization details (fallback if pynvml not available)
        gpu_utilization = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # Try to get GPU utilization if nvidia-ml-py is available
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization.append({
                        "device": i,
                        "gpu_util": util.gpu,
                        "memory_util": util.memory
                    })
                except ImportError:
                    # Fallback if pynvml not available
                    gpu_utilization.append({
                        "device": i,
                        "gpu_util": None,
                        "memory_util": None,
                        "note": "pynvml not available for detailed GPU stats"
                    })
        
        return {
            "status": "healthy",
            "uptime_seconds": uptime.total_seconds(),
            "uptime_formatted": str(uptime),
            "process": {
                "pid": process.pid,
                "memory_info": process.memory_info()._asdict(),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads()
            },
            "gpu_utilization": gpu_utilization,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            "network": {
                "connections": len(psutil.net_connections()),
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Root endpoint for basic API information
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Deforum Alpha API",
        "version": "1.0.0",
        "description": "Interactive API for Deforum Flux video generation",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "system_status": "/api/v1/system/status",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "environment": "runpod" if os.getenv("RUNPOD_MODE") else "local"
    }

# Favicon endpoint to prevent 404 errors
@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404 errors"""
    from fastapi.responses import Response
    return Response(status_code=204)

# API routes
app.include_router(generation_router, prefix="/api/v1", tags=["generation"])
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])
app.include_router(presets_router, prefix="/api/v1", tags=["presets"])
app.include_router(config_router, prefix="/api/v1/config", tags=["configuration"])
app.include_router(animation_router, prefix="/api/v1/animation", tags=["animation"])
app.include_router(payload_router, prefix="/api/v1/payload", tags=["unified-payload"])
app.include_router(models_router, prefix="/api/v1", tags=["models"])
app.include_router(models_management_router, prefix="/api/v1", tags=["models-management"])

# Model routes are now consolidated into basic and management modules

# Model health and testing routes
if MODEL_HEALTH_AVAILABLE:
    app.include_router(model_health_router, prefix="/api/v1", tags=["model-health"])
    logger.info("Model health and testing routes enabled")

# Serve frontend static files in production (RunPod)
if os.getenv("RUNPOD_MODE") or os.getenv("PRODUCTION"):
    frontend_path = Path(__file__).parent.parent / "frontend" / "out"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
        logger.info(f"Serving frontend from {frontend_path}")

if __name__ == "__main__":
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "7860"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        log_level="info",
        reload=not bool(os.getenv("PRODUCTION"))
    )
