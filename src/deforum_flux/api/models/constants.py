"""
Model Constants and Configurations
==================================

Single source of truth for model definitions, constants, and configurations
used throughout the API.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class ModelStatus(Enum):
    """Model availability status."""
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed" 
    PARTIAL = "partial"
    INSTALLING = "installing"
    ERROR = "error"


class ModelType(Enum):
    """Model type classification."""
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    FLUX_FILL = "flux-dev-fill"
    FLUX_CANNY = "flux-dev-canny"


@dataclass
class ModelInfo:
    """Model information structure."""
    id: str
    name: str
    description: str
    memory_requirements: str
    size_gb: float
    type: ModelType
    recommended: bool = False
    status: ModelStatus = ModelStatus.AVAILABLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "memory_requirements": self.memory_requirements,
            "size_gb": self.size_gb,
            "type": self.type.value,
            "recommended": self.recommended,
            "status": self.status.value
        }


# Static model definitions - Single source of truth
AVAILABLE_MODELS = [
    ModelInfo(
        id="flux-dev",
        name="FLUX.1-dev",
        description="High-quality image generation model with excellent detail and coherence",
        memory_requirements="24GB+",
        size_gb=19.8,
        type=ModelType.FLUX_DEV,
        recommended=True,
        status=ModelStatus.AVAILABLE
    ),
    ModelInfo(
        id="flux-schnell",
        name="FLUX.1-schnell", 
        description="Fast image generation model optimized for speed",
        memory_requirements="16GB+",
        size_gb=14.9,
        type=ModelType.FLUX_SCHNELL,
        recommended=False,
        status=ModelStatus.AVAILABLE
    ),
    ModelInfo(
        id="flux-fill",
        name="FLUX.1-fill",
        description="Inpainting model for filling masked regions",
        memory_requirements="20GB+",
        size_gb=17.2,
        type=ModelType.FLUX_FILL,
        recommended=False,
        status=ModelStatus.NOT_INSTALLED
    ),
    ModelInfo(
        id="flux-canny",
        name="FLUX.1-canny",
        description="ControlNet model for edge-guided generation",
        memory_requirements="22GB+", 
        size_gb=18.5,
        type=ModelType.FLUX_CANNY,
        recommended=False,
        status=ModelStatus.NOT_INSTALLED
    )
]

# Model lookup by ID
MODELS_BY_ID = {model.id: model for model in AVAILABLE_MODELS}

# Default model configuration
DEFAULT_MODEL_ID = "flux-dev"
FALLBACK_MODEL_ID = "flux-schnell"

# API Configuration Constants
API_CONSTANTS = {
    "max_models_per_page": 50,
    "default_page_size": 10,
    "supported_formats": ["safetensors", "ckpt", "pt"],
    "max_model_name_length": 100,
    "max_description_length": 500
}

# Model validation rules
VALIDATION_RULES = {
    "min_memory_gb": 8,
    "max_memory_gb": 80,
    "min_size_gb": 0.1,
    "max_size_gb": 100.0,
    "valid_statuses": [status.value for status in ModelStatus],
    "valid_types": [model_type.value for model_type in ModelType]
}

# Backend integration constants
BACKEND_CONFIG = {
    "flux_util_timeout": 30,
    "installation_check_interval": 5,
    "max_retry_attempts": 3,
    "backend_health_check_timeout": 10
}


def get_model_by_id(model_id: str) -> ModelInfo:
    """
    Get model info by ID.
    
    Args:
        model_id: Model identifier
        
    Returns:
        ModelInfo object
        
    Raises:
        KeyError: If model ID not found
    """
    if model_id not in MODELS_BY_ID:
        raise KeyError(f"Model '{model_id}' not found")
    return MODELS_BY_ID[model_id]


def get_available_models() -> List[ModelInfo]:
    """Get list of all available models."""
    return AVAILABLE_MODELS.copy()


def get_models_by_status(status: ModelStatus) -> List[ModelInfo]:
    """Get models filtered by status."""
    return [model for model in AVAILABLE_MODELS if model.status == status]


def get_recommended_models() -> List[ModelInfo]:
    """Get recommended models."""
    return [model for model in AVAILABLE_MODELS if model.recommended]


def validate_model_id(model_id: str) -> bool:
    """Validate if model ID exists."""
    return model_id in MODELS_BY_ID


def get_model_stats() -> Dict[str, Any]:
    """Get statistics about available models."""
    total = len(AVAILABLE_MODELS)
    by_status = {}
    by_type = {}
    
    for model in AVAILABLE_MODELS:
        # Count by status
        status_key = model.status.value
        by_status[status_key] = by_status.get(status_key, 0) + 1
        
        # Count by type
        type_key = model.type.value
        by_type[type_key] = by_type.get(type_key, 0) + 1
    
    return {
        "total_models": total,
        "by_status": by_status,
        "by_type": by_type,
        "recommended_count": len(get_recommended_models()),
        "total_size_gb": sum(model.size_gb for model in AVAILABLE_MODELS)
    } 