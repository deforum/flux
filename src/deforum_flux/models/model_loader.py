"""
Model loader for Deforum Flux animations.

Leverages flux.util directly for model loading and provides optional TRT optimization.
Caching and production-ready inference.
"""

import os
from typing import Dict, Optional, Any, Tuple
import torch

from flux.util import load_flow_model, load_t5, load_clip, load_ae, configs, check_onnx_access_for_trt

# Make TRT imports optional
try:
    from flux.trt.trt_manager import TRTManager, ModuleName
    TRT_AVAILABLE = True
except ImportError as e:
    TRT_AVAILABLE = False
    TRTManager = None
    ModuleName = None

from deforum.core.exceptions import ModelLoadingError, FluxModelError
from deforum.core.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Model loader for Deforum Flux animations.
    
    Features:
    - Direct flux.util loading for standard models
    - Optional TRT optimization for production inference (if available)
    - Model caching by (model_name, device) key
    - Error handling with Deforum exceptions
    """
    
    def __init__(self):
        """Initialize model loader with empty cache."""
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self._trt_manager: Optional[Any] = None
        
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available - TRT optimizations will be disabled")
    
    def load_models(
        self,
        model_name: str,
        device: str = "cuda",
        use_trt: bool = False,
        trt_precision: str = "bf16"
    ) -> Dict[str, Any]:
        """
        Load Flux models for animation inference.
        
        Args:
            model_name: Flux model name (e.g., "flux-dev", "flux-schnell")
            device: Target device ("cuda", "cpu")
            use_trt: Enable TensorRT optimization for production (requires TRT)
            trt_precision: TRT precision ("bf16", "fp8", "fp4")
            
        Returns:
            Dictionary containing loaded models:
            {
                "model": flux_model,
                "ae": autoencoder,
                "t5": t5_encoder,
                "clip": clip_encoder,
                "trt_engines": optional_trt_engines
            }
            
        Raises:
            ModelLoadingError: If model loading fails
            FluxModelError: If model configuration is invalid
        """
        # Check TRT availability
        if use_trt and not TRT_AVAILABLE:
            logger.warning("TRT requested but not available - falling back to standard loading")
            use_trt = False
        
        # Create cache key
        cache_key = f"{model_name}_{device}{'_trt' if use_trt else ''}"
        
        # Return cached models if available
        if cache_key in self._model_cache:
            logger.info(f"Returning cached models for {cache_key}")
            return self._model_cache[cache_key]
        
        logger.info(f"Loading Flux models: {model_name} on {device}")
        
        try:
            # Validate model name
            if model_name not in configs:
                available_models = list(configs.keys())
                raise FluxModelError(
                    f"Unknown model '{model_name}'",
                    model_name=model_name,
                    available_models=available_models
                )
            
            # Load models using flux utilities
            models = self._load_standard_models(model_name, device)
            
            # Add TRT optimization if requested and available
            if use_trt and TRT_AVAILABLE:
                models["trt_engines"] = self._load_trt_engines(
                    model_name, device, trt_precision
                )
            
            # Cache the loaded models
            self._model_cache[cache_key] = models
            
            logger.info(f"Successfully loaded models for {model_name}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to load models for {model_name}: {e}")
            raise ModelLoadingError(
                f"Failed to load {model_name} models",
                model_name=model_name,
                device=device,
                original_error=e
            ) from e
    
    def _load_standard_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """Load standard Flux models using flux.util functions."""
        logger.info(f"Loading standard models for {model_name}")
        
        models = {}
        
        # Load main Flux model
        logger.info("Loading Flux transformer model...")
        models["model"] = load_flow_model(model_name, device=device)
        
        # Load autoencoder
        logger.info("Loading autoencoder...")
        models["ae"] = load_ae(model_name, device=device)
        
        # Load text encoders
        logger.info("Loading T5 text encoder...")
        models["t5"] = load_t5(device=device)
        
        logger.info("Loading CLIP text encoder...")
        models["clip"] = load_clip(device=device)
        
        return models
    
    def _load_trt_engines(
        self,
        model_name: str,
        device: str,
        precision: str = "bf16"
    ) -> Optional[Dict[Any, Any]]:
        """Load TensorRT optimized engines for production inference."""
        if not TRT_AVAILABLE:
            logger.warning("TRT not available - skipping TRT engine loading")
            return None
            
        logger.info(f"Loading TRT engines for {model_name} with {precision} precision")
        
        try:
            # Check if ONNX models are available for TRT
            custom_onnx_paths = check_onnx_access_for_trt(model_name, precision)
            if not custom_onnx_paths:
                logger.warning(f"No ONNX models available for TRT optimization of {model_name}")
                return None
            
            # Initialize TRT manager if not already done
            if self._trt_manager is None:
                self._trt_manager = TRTManager(
                    trt_transformer_precision=precision,
                    trt_t5_precision="bf16",  # T5 typically uses bf16
                    max_batch=2,
                    verbose=True
                )
            
            # Define modules to optimize
            module_names = {
                ModuleName.CLIP,
                ModuleName.TRANSFORMER,
                ModuleName.T5,
                ModuleName.VAE,
                ModuleName.VAE_ENCODER
            }
            
            # Set up TRT engine directory
            engine_dir = os.path.join(os.environ.get("TRT_ENGINE_DIR", "checkpoints/trt_engines"), model_name)
            
            # Load TRT engines
            engines = self._trt_manager.load_engines(
                model_name=model_name,
                module_names=module_names,
                engine_dir=engine_dir,
                custom_onnx_paths=custom_onnx_paths,
                trt_image_height=1024,  # Default height for animations
                trt_image_width=1024,   # Default width for animations
                trt_batch_size=1
            )
            
            logger.info(f"Successfully loaded {len(engines)} TRT engines")
            return engines
            
        except Exception as e:
            logger.error(f"Failed to load TRT engines for {model_name}: {e}")
            # Don't fail the entire loading process for TRT issues
            return None
    
    def get_trt_manager(self, model_name: str, precision: str = "bf16") -> Optional[Any]:
        """
        Get TRT manager for advanced TRT operations.
        
        Args:
            model_name: Model name for TRT configuration
            precision: TRT precision setting
            
        Returns:
            TRTManager instance or None if TRT not available
        """
        if not TRT_AVAILABLE:
            logger.warning("TRT not available")
            return None
            
        if self._trt_manager is None:
            try:
                self._trt_manager = TRTManager(
                    trt_transformer_precision=precision,
                    trt_t5_precision="bf16",
                    max_batch=2,
                    verbose=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize TRT manager: {e}")
                return None
        
        return self._trt_manager
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear model cache to free memory.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name:
            # Clear specific model variants
            keys_to_remove = [key for key in self._model_cache.keys() if key.startswith(model_name)]
            for key in keys_to_remove:
                del self._model_cache[key]
            logger.info(f"Cleared cache for {model_name}")
        else:
            # Clear all cached models
            self._model_cache.clear()
            logger.info("Cleared all model cache")
        
        # Force garbage collection and CUDA cache clearing
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cached_models(self) -> Dict[str, bool]:
        """
        Get status of cached models.
        
        Returns:
            Dictionary mapping cache keys to True (indicating cached)
        """
        return {key: True for key in self._model_cache.keys()}
    
    def estimate_memory_usage(self, model_name: str) -> Dict[str, str]:
        """
        Estimate memory usage for a model.
        
        Args:
            model_name: Name of the model to estimate
            
        Returns:
            Dictionary with memory estimates for each component
        """
        if model_name not in configs:
            return {"error": "Unknown model"}
        
        # Rough estimates based on model parameters
        config = configs[model_name]
        
        # Estimate based on hidden size and depth
        hidden_size = config.params.hidden_size
        depth = config.params.depth
        
        # Very rough estimates in GB
        model_gb = (hidden_size * depth * 4) / (1024**3)  # Rough parameter count estimation
        ae_gb = 0.5  # AE is relatively small
        t5_gb = 4.0  # T5-XXL is large
        clip_gb = 0.5  # CLIP is relatively small
        
        return {
            "flux_model": f"~{model_gb:.1f}GB",
            "autoencoder": f"~{ae_gb:.1f}GB",
            "t5_encoder": f"~{t5_gb:.1f}GB", 
            "clip_encoder": f"~{clip_gb:.1f}GB",
            "total_estimate": f"~{model_gb + ae_gb + t5_gb + clip_gb:.1f}GB",
            "trt_available": str(TRT_AVAILABLE)
        }
    
    @property
    def trt_available(self) -> bool:
        """Check if TRT is available."""
        return TRT_AVAILABLE


# Create a global instance for easy access
model_loader = ModelLoader()
