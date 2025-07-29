"""
Deforum Flux Models Package

Provides clean, simple model loading for Flux animations.
Now uses simplified model management with flux.util directly.
"""

from .model_loader import ModelLoader, model_loader
from .models import (
    ModelManager, ModelInfo, ModelSet,
    get_model_manager, setup_models_for_backend, get_models, 
    initialize_models, download_model, download_onnx_model,
    get_available_models
)

__all__ = [
    "ModelLoader", 
    "model_loader",
    "ModelManager",
    "ModelInfo",
    "ModelSet", 
    "get_model_manager",
    "setup_models_for_backend",
    "get_models",
    "initialize_models",
    "download_model",
    "download_onnx_model",
    "get_available_models"
]

# Compatibility aliases for old imports
from . import models as model_manager  # Support "from deforum_flux.models import model_manager"
