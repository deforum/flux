"""
Basic Models API Routes - SIMPLIFIED VERSION
===========================================

Core model endpoints for listing, status checking, and basic information.
Now uses simplified model management with flux.util directly.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import logging

# Import our centralized model constants
from deforum_flux.api.models.constants import (
    ModelInfo, ModelStatus, 
    get_available_models, get_model_by_id, get_model_stats,
    validate_model_id, DEFAULT_MODEL_ID
)

# Simplified backend integration
try:
    from deforum_flux.models.models import get_model_manager, ModelManager
    BACKEND_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Simplified model management not available: {e}")
    BACKEND_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

# Global state
current_model_id = DEFAULT_MODEL_ID
_model_manager: Optional[ModelManager] = None


def get_backend_model_manager() -> Optional[ModelManager]:
    """Get or create the simplified model manager instance."""
    global _model_manager
    
    if not BACKEND_AVAILABLE:
        return None
    
    if _model_manager is None:
        try:
            _model_manager = get_model_manager()
        except Exception as e:
            logger.error(f"Failed to initialize simplified model manager: {e}")
            return None
    
    return _model_manager


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """
    List all available models with their status and capabilities.
    
    Returns:
        Dict containing available models and current model information
    """
    try:
        logger.info("Listing available models")
        
        # Try to get models from simplified backend if available
        if BACKEND_AVAILABLE:
            manager = get_backend_model_manager()
            if manager and manager.available:
                try:
                    # Get model sets from simplified backend
                    model_sets = manager.get_flux_model_sets()
                    installation_status = manager.get_installation_status()
                    
                    available_models = []
                    for set_name, model_set in model_sets.items():
                        status_info = installation_status.get(set_name, {})
                        
                        # Determine status based on installation
                        if status_info.get("is_complete", False):
                            status = "installed"
                        elif status_info.get("installed_models", 0) > 0:
                            status = "partial"
                        else:
                            status = "not_installed"
                        
                        available_models.append({
                            "id": set_name,
                            "name": model_set.name,
                            "status": status,
                            "description": model_set.description,
                            "memory_requirements": f"{model_set.recommended_gpu_memory_gb}GB+",
                            "size_gb": model_set.total_size_gb,
                            "installed_models": status_info.get("installed_models", 0),
                            "total_models": status_info.get("total_models", len(model_set.models)),
                            "recommended": set_name == "flux-dev"
                        })
                    
                    return {
                        "available_models": available_models,
                        "current_model": current_model_id,
                        "total_models": len(available_models),
                        "backend_available": True,
                        "status": "success"
                    }
                except Exception as e:
                    logger.warning(f"Simplified backend model listing failed: {e}")
        
        # Fallback to static model list from constants
        static_models = get_available_models()
        model_dicts = [model.to_dict() for model in static_models]
        
        return {
            "available_models": model_dicts,
            "current_model": current_model_id,
            "total_models": len(model_dicts),
            "backend_available": False,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/status")
async def get_models_status() -> Dict[str, Any]:
    """
    Get overall model status and readiness information - SIMPLIFIED VERSION
    
    Returns:
        Dict containing models ready count and status information
    """
    try:
        logger.info("Getting overall models status")
        
        if not BACKEND_AVAILABLE:
            # Use static model data for status
            static_stats = get_model_stats()
            available_count = static_stats["by_status"].get("available", 0)
            
            return {
                "backend_available": False,
                "models_ready": available_count,
                "total_models": static_stats["total_models"],
                "ready_models": [m.id for m in get_available_models() if m.status == ModelStatus.AVAILABLE],
                "status": "static_fallback",
                "message": f"{available_count}/{static_stats['total_models']} models available (static)"
            }
        
        manager = get_backend_model_manager()
        if not manager or not manager.available:
            return {
                "backend_available": False,
                "models_ready": 0,
                "total_models": 0,
                "ready_models": [],
                "status": "manager_unavailable",
                "message": "Simplified model manager not available"
            }
        
        # Get installation status for all models from simplified backend
        try:
            model_sets = manager.get_flux_model_sets()
            installation_status = manager.get_installation_status()
            
            models_ready = 0
            total_models = len(model_sets)
            
            ready_models = []
            partial_models = []
            missing_models = []
            
            for set_name, model_set in model_sets.items():
                status_info = installation_status.get(set_name, {})
                
                if status_info.get("is_complete", False):
                    models_ready += 1
                    ready_models.append(set_name)
                elif status_info.get("installed_models", 0) > 0:
                    partial_models.append(set_name)
                else:
                    missing_models.append(set_name)
            
            return {
                "backend_available": True,
                "models_ready": models_ready,
                "total_models": total_models,
                "ready_models": ready_models,
                "partial_models": partial_models,
                "missing_models": missing_models,
                "status": "success" if models_ready > 0 else "no_models_ready",
                "message": f"{models_ready}/{total_models} models ready"
            }
        
        except Exception as e:
            logger.warning(f"Simplified backend model status check failed: {e}")
            return {
                "backend_available": True,
                "models_ready": 0,
                "total_models": 0,
                "ready_models": [],
                "partial_models": [],
                "missing_models": [],
                "status": "backend_error",
                "message": f"Backend error: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        return {
            "backend_available": BACKEND_AVAILABLE,
            "models_ready": 0,
            "total_models": 0,
            "ready_models": [],
            "status": "error",
            "message": f"Status check failed: {str(e)}"
        }

@router.get("/models/stats")
async def get_models_statistics() -> Dict[str, Any]:
    """
    Get comprehensive statistics about available models.
    
    Returns:
        Dict containing model statistics and summaries
    """
    try:
        logger.info("Getting model statistics")
        
        # Get base statistics from constants
        base_stats = get_model_stats()
        
        # Try to enhance with simplified backend statistics
        if BACKEND_AVAILABLE:
            manager = get_backend_model_manager()
            if manager and manager.available:
                try:
                    backend_stats = manager.get_installation_status()
                    base_stats["backend_stats"] = {
                        "available": True,
                        "managed_models": len(backend_stats),
                        "installation_status": backend_stats
                    }
                except Exception as e:
                    logger.warning(f"Could not get simplified backend statistics: {e}")
        
        base_stats["backend_available"] = BACKEND_AVAILABLE
        base_stats["status"] = "success"
        
        return base_stats
        
    except Exception as e:
        logger.error(f"Error getting model statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model statistics: {str(e)}"
        )


@router.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: The ID of the model to get information for
        
    Returns:
        Dict containing detailed model information
    """
    try:
        # First try to get from constants (always available)
        if validate_model_id(model_id):
            model_info = get_model_by_id(model_id)
            base_info = model_info.to_dict()
            
            # Try to enhance with simplified backend information if available
            if BACKEND_AVAILABLE:
                manager = get_backend_model_manager()
                if manager and manager.available:
                    try:
                        model_sets = manager.get_flux_model_sets()
                        if model_id in model_sets:
                            model_set = model_sets[model_id]
                            installation_status = manager.get_installation_status().get(model_id, {})
                            
                            # Enhanced info from simplified backend
                            base_info.update({
                                "backend_info": {
                                    "installed_models": installation_status.get("installed_models", 0),
                                    "total_models": installation_status.get("total_models", 0),
                                    "is_complete": installation_status.get("is_complete", False),
                                    "total_size_gb": model_set.total_size_gb,
                                    "recommended_gpu_memory_gb": model_set.recommended_gpu_memory_gb
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Could not get simplified backend info for {model_id}: {e}")
            
            logger.info(f"Retrieved model info for: {model_id}")
            return {
                "model": base_info,
                "backend_available": BACKEND_AVAILABLE,
                "status": "success"
            }
        
        # Model not found
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_id}' not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

@router.get("/models/enhanced/installation-status")
async def get_enhanced_models_status() -> Dict[str, Any]:
    """
    Get enhanced model installation status - Simplified version
    
    Returns:
        Enhanced installation status information
    """
    try:
        logger.info("Getting enhanced models installation status")
        
        if not BACKEND_AVAILABLE:
            return {
                "available": False,
                "message": "Enhanced model management not available",
                "backend_available": False,
                "installation_status": {},
                "status": "unavailable"
            }
        
        manager = get_backend_model_manager()
        if not manager or not manager.available:
            return {
                "available": False,
                "message": "Simplified model manager could not be initialized",
                "backend_available": True,
                "installation_status": {},
                "status": "manager_error"
            }
        
        try:
            installation_status = manager.get_installation_status()
            model_sets = manager.get_flux_model_sets()
            
            enhanced_status = {}
            for set_name, status_info in installation_status.items():
                model_set = model_sets.get(set_name)
                enhanced_status[set_name] = {
                    **status_info,
                    "model_set_info": {
                        "name": model_set.name if model_set else set_name,
                        "description": model_set.description if model_set else "",
                        "total_size_gb": model_set.total_size_gb if model_set else 0,
                        "recommended_gpu_memory_gb": model_set.recommended_gpu_memory_gb if model_set else 0
                    }
                }
            
            return {
                "available": True,
                "message": "Enhanced model status retrieved successfully",
                "backend_available": True,
                "installation_status": enhanced_status,
                "total_model_sets": len(enhanced_status),
                "status": "success"
            }
        
        except Exception as e:
            logger.warning(f"Enhanced status retrieval failed: {e}")
            return {
                "available": True,
                "message": f"Enhanced status retrieval failed: {str(e)}",
                "backend_available": True,
                "installation_status": {},
                "status": "error"
            }
        
    except Exception as e:
        logger.error(f"Error getting enhanced models status: {e}")
        return {
            "available": False,
            "message": f"Enhanced models status failed: {str(e)}",
            "backend_available": BACKEND_AVAILABLE,
            "installation_status": {},
            "status": "error"
        }
