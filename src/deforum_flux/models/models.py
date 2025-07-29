"""
Core Models Module for Deforum Backend

This module provides model management using flux.util directly.

"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Import flux.util functions directly
try:
    from flux.util import configs, get_checkpoint_path, download_onnx_models_for_trt
    FLUX_UTIL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"flux.util not available: {e}")
    FLUX_UTIL_AVAILABLE = False
    configs = {}

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """ Model information structure."""
    id: str
    name: str
    description: str
    size_gb: float
    memory_requirements: str
    status: str = "available"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "size_gb": self.size_gb,
            "memory_requirements": self.memory_requirements,
            "status": self.status
        }

@dataclass 
class ModelSet:
    """ Model set structure."""
    name: str
    description: str
    models: List[str]
    total_size_gb: float
    recommended_gpu_memory_gb: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "models": self.models,
            "total_size_gb": self.total_size_gb,
            "recommended_gpu_memory_gb": self.recommended_gpu_memory_gb
        }

class ModelManager:
    """Model manager using flux.util directly."""
    
    def __init__(self):
        self.available = FLUX_UTIL_AVAILABLE
        
    def get_flux_model_sets(self) -> Dict[str, ModelSet]:
        """Get available Flux model sets from flux.util configs."""
        if not self.available:
            return {}
            
        model_sets = {}
        for model_name, config in configs.items():
            # Create simple model set from flux config
            model_set = ModelSet(
                name=model_name.replace("-", " ").title(),
                description=f"Flux model: {model_name}",
                models=[config.repo_flow, config.repo_ae],
                total_size_gb=self._estimate_size_gb(model_name),
                recommended_gpu_memory_gb=self._get_memory_requirement(model_name)
            )
            model_sets[model_name] = model_set
            
        return model_sets
    
    def get_installation_status(self) -> Dict[str, Dict[str, Any]]:
        """Get installation status for all models."""
        if not self.available:
            return {}
            
        status = {}
        for model_name in configs.keys():
            try:
                # Check if model files exist by trying to get their paths
                config = configs[model_name]
                flow_path = get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL")
                ae_path = get_checkpoint_path(config.repo_id, config.repo_ae, "FLUX_AE")
                
                # Count existing files
                installed_models = 0
                total_models = 2  # flow + ae
                
                if flow_path.exists():
                    installed_models += 1
                if ae_path.exists():
                    installed_models += 1
                    
                # Check for LoRA if applicable
                if hasattr(config, 'lora_repo_id') and config.lora_repo_id:
                    total_models += 1
                    lora_path = get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA")
                    if lora_path.exists():
                        installed_models += 1
                
                status[model_name] = {
                    "installed_models": installed_models,
                    "total_models": total_models,
                    "is_complete": installed_models == total_models,
                    "status": "complete" if installed_models == total_models else "partial" if installed_models > 0 else "missing"
                }
                
            except Exception as e:
                logger.warning(f"Could not check status for {model_name}: {e}")
                status[model_name] = {
                    "installed_models": 0,
                    "total_models": 2,
                    "is_complete": False,
                    "status": "error"
                }
                
        return status
    
    def _estimate_size_gb(self, model_name: str) -> float:
        """Estimate model size in GB."""
        size_estimates = {
            "flux-dev": 19.8,
            "flux-schnell": 14.9,
            "flux-dev-canny": 18.5,
            "flux-dev-depth": 18.5,
            "flux-dev-fill": 17.2,
            "flux-dev-redux": 18.0,
            "flux-dev-kontext": 19.0
        }
        return size_estimates.get(model_name, 15.0)
    
    def _get_memory_requirement(self, model_name: str) -> int:
        """Get memory requirement in GB."""
        memory_requirements = {
            "flux-dev": 24,
            "flux-schnell": 16,
            "flux-dev-canny": 22,
            "flux-dev-depth": 22,
            "flux-dev-fill": 20,
            "flux-dev-redux": 22,
            "flux-dev-kontext": 24
        }
        return memory_requirements.get(model_name, 20)

# Global simple model manager instance
_global_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Get the global simple model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager

def setup_models_for_backend(models_path: Optional[str] = None) -> ModelManager:
    """Setup models for backend - simplified version."""
    global _global_model_manager
    _global_model_manager = ModelManager()
    return _global_model_manager

def get_models() -> ModelManager:
    """Get the global model manager instance."""
    return get_model_manager()

def initialize_models(models_path: Optional[str] = None) -> ModelManager:
    """Initialize the models system."""
    return setup_models_for_backend(models_path)

def download_model(model_name: str) -> bool:
    """Download a specific model using flux.util."""
    if not FLUX_UTIL_AVAILABLE:
        logger.error("flux.util not available for model download")
        return False
        
    if model_name not in configs:
        logger.error(f"Unknown model: {model_name}")
        return False
        
    try:
        config = configs[model_name]
        
        # Download main model files
        get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL")
        get_checkpoint_path(config.repo_id, config.repo_ae, "FLUX_AE")
        
        # Download LoRA if applicable
        if hasattr(config, 'lora_repo_id') and config.lora_repo_id:
            get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA")
            
        logger.info(f"Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False

def download_onnx_model(model_name: str, precision: str = "bf16") -> Optional[str]:
    """Download ONNX models for TRT using flux.util."""
    if not FLUX_UTIL_AVAILABLE:
        logger.error("flux.util not available for ONNX download")
        return None
        
    try:
        return download_onnx_models_for_trt(model_name, precision)
    except Exception as e:
        logger.error(f"Failed to download ONNX models for {model_name}: {e}")
        return None

def get_available_models() -> List[str]:
    """Get list of available model names."""
    if not FLUX_UTIL_AVAILABLE:
        return []
    return list(configs.keys())


# Export main classes for easy access
__all__ = [
    'ModelManager',
    'ModelInfo', 
    'ModelSet',
    'get_model_manager',
    'setup_models_for_backend',
    'get_models',
    'initialize_models',
    'download_model',
    'download_onnx_model',
    'get_available_models'
]
