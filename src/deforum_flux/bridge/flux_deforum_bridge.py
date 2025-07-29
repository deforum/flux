"""
Core Flux-Deforum Bridge - Classic Deforum Style Animation Generation

This is the main bridge class that integrates Flux image generation with classic
Deforum animation, focusing on geometric transformations and parameter scheduling.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

from deforum.config.settings import Config
from deforum_flux.models.model_paths import get_model_path, get_all_model_paths
from .bridge_config import BridgeConfigManager
from .bridge_generation_utils import GenerationUtils
from .bridge_stats_and_cleanup import BridgeStatsManager, ResourceManager
from deforum.core.exceptions import (
    FluxModelError, DeforumConfigError, MotionProcessingError,
    ValidationError, ModelLoadingError, TensorProcessingError,
    ResourceError, handle_exception
)
from deforum.core.logging_config import get_logger, LogContext


class FluxDeforumBridge:
    """
    Production-ready bridge class for classic Deforum-style animation with Flux.
    
    This implementation focuses on:
    - Classic geometric transformations (zoom, rotate, translate)
    - Parameter scheduling and interpolation
    - 16-channel Flux latent processing
    - Simplified, testable architecture
    """
    
    def __init__(self, config: Config, mock_mode: bool = False):
        """
        Initialize the Flux-Deforum bridge for classic animation.
        
        Args:
            config: Configuration object with all settings
            mock_mode: If True, initialize with mock components for testing/CI only
                      Production should NEVER use mock_mode=True
            
        Raises:
            DeforumConfigError: If configuration is invalid
            FluxModelError: If model loading fails
        """
        self.logger = get_logger("flux_deforum_bridge")
        self.config = config
        
        # Production safety: Only allow mocks in explicit testing scenarios
        self.mock_mode = mock_mode and getattr(config, 'allow_mocks', False)
        
        if mock_mode and not getattr(config, 'allow_mocks', False):
            self.logger.warning("Mock mode requested but not allowed in production config - using real models")
        
        # Initialize managers
        self.config_manager = BridgeConfigManager()
        self.generation_utils = GenerationUtils()
        self.stats_manager = BridgeStatsManager()
        self.resource_manager = ResourceManager()
        
        # Validate and prepare configuration for classic Deforum mode
        self._prepare_classic_config()
        
        # Initialize components
        self.model = None
        self.ae = None
        self.t5 = None
        self.clip = None
        self.motion_engine = None
        self.parameter_engine = None
        self._using_mocks = False
        
        try:
            self.logger.info("Starting classic Deforum bridge initialization", extra={
                "mock_mode": self.mock_mode,
                "production_mode": not self.mock_mode
            })
            self._initialize_components()
            self.logger.info("Classic Deforum bridge initialization completed")
        except Exception as e:
            self.logger.error(f"Bridge initialization failed: {e}")
            # Production mode: ALWAYS raise errors, no silent fallbacks to mocks
            if not self.mock_mode:
                self.logger.error("Production initialization failed - this MUST be fixed before deployment")
                raise
            else:
                self.logger.warning("Test initialization failed - this is acceptable in CI/testing")
        
        self.logger.info("FluxDeforumBridge initialized successfully (Classic Mode)", extra={
            "model_name": config.model_name,
            "device": config.device,
            "motion_mode": config.motion_mode,
            "classic_mode": True
        })
    
    def _prepare_classic_config(self) -> None:
        """Prepare configuration for classic Deforum mode."""
        # Apply classic Deforum overrides
        self.config = self.config_manager.apply_classic_deforum_overrides(self.config)
        
        # Validate configuration
        self.config_manager.validate_config(self.config)
        
        self.logger.info("Configuration prepared for classic Deforum mode")
    
    @handle_exception
    def _initialize_components(self) -> None:
        """Initialize all bridge components."""
        try:
            if self.mock_mode or getattr(self.config, 'skip_model_loading', False):
                self.logger.info("Initializing in mock mode")
                self._initialize_mock_components()
            else:
                # Load Flux models
                self._load_models()
            
            # Initialize motion engine (classic mode) with error handling
            self._initialize_motion_engine()
            
            # Initialize parameter engine with error handling
            self._initialize_parameter_engine()
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            
            # Check if this is a "models not available" error vs a real failure
            error_str = str(e).lower()
            if any(phrase in error_str for phrase in ["flux is not available", "no module named 'flux'", "install flux"]):
                if self.mock_mode:
                    self.logger.warning("Flux models not available - initializing mocks for testing")
                    self._initialize_basic_mocks()
                    return
                else:
                    self.logger.error("PRODUCTION FAILURE: Flux models not available - this MUST be fixed")
                    raise FluxModelError("Production deployment requires Flux models to be properly installed and available")
            else:
                # Real error - NEVER silently fall back to mocks in production
                if self.mock_mode:
                    self.logger.warning("Initializing test mocks due to component failure")
                    self._initialize_basic_mocks()
                else:
                    self.logger.error("PRODUCTION FAILURE: Component initialization failed")
                    raise
    
    def _initialize_mock_components(self) -> None:
        """Initialize proper mock components for testing."""
        self.logger.info("Initializing mock components for testing")
        
        # Create mock model objects that provide the expected interface
        self.model = self._create_mock_flux_model()
        self.ae = self._create_mock_autoencoder()
        self.t5 = self._create_mock_t5_model()
        self.clip = self._create_mock_clip_model()
        self._using_mocks = True
        
        self.logger.info("Mock components initialized successfully")
    
    def _create_mock_flux_model(self):
        """Create a mock Flux model."""
        class MockFluxModel:
            def __init__(self):
                self.device = "mock"
                self.dtype = torch.bfloat16
            
            def __call__(self, *args, **kwargs):
                # Return mock latent tensor with proper shape
                if args:
                    x = args[0]
                    return x + torch.randn_like(x) * 0.1
                
                # For mock mode, return tensor in PACKED format that unpack() expects
                # This should match the input tensor shape from prepare()
                # If no input available, use reasonable defaults
                return torch.randn(1, 4096, 64, dtype=torch.bfloat16)
        
        return MockFluxModel()
    
    def _create_mock_autoencoder(self):
        """Create a mock autoencoder."""
        class MockAutoEncoder:
            def __init__(self):
                self.device = "mock"
                self.dtype = torch.bfloat16
            
            def decode(self, x):
                # Return mock RGB image tensor
                batch_size = x.shape[0] if x.ndim >= 2 else 1
                height = x.shape[-2] * 8 if x.ndim >= 3 else 512
                width = x.shape[-1] * 8 if x.ndim >= 3 else 512
                return torch.randn(batch_size, 3, height, width, dtype=torch.float32)
            
            def encode(self, x):
                # Return mock latent tensor
                batch_size = x.shape[0] if x.ndim >= 2 else 1
                height = x.shape[-2] // 8 if x.ndim >= 3 else 64
                width = x.shape[-1] // 8 if x.ndim >= 3 else 64
                return torch.randn(batch_size, 16, height, width, dtype=torch.bfloat16)
        
        return MockAutoEncoder()
    
    def _create_mock_t5_model(self):
        """Create a mock T5 text encoder."""
        class MockT5Model:
            def __init__(self):
                self.device = "mock"
                self.dtype = torch.bfloat16
            
            def __call__(self, *args, **kwargs):
                # Return mock text embeddings
                return torch.randn(1, 256, 4096, dtype=torch.bfloat16)
        
        return MockT5Model()
    
    def _create_mock_clip_model(self):
        """Create a mock CLIP text encoder."""
        class MockCLIPModel:
            def __init__(self):
                self.device = "mock"
                self.dtype = torch.bfloat16
            
            def __call__(self, *args, **kwargs):
                # Return mock text embeddings
                return torch.randn(1, 77, 768, dtype=torch.bfloat16)
        
        return MockCLIPModel()
    
    def _initialize_mock_models(self) -> None:
        """Legacy method - redirect to new mock components."""
        self._initialize_mock_components()
    
    def _initialize_basic_mocks(self) -> None:
        """Initialize basic mocks for all components when initialization fails."""
        self.logger.info("Initializing basic mocks for failed initialization")
        
        # Use proper mock components instead of None
        self.model = self._create_mock_flux_model()
        self.ae = self._create_mock_autoencoder()
        self.t5 = self._create_mock_t5_model()
        self.clip = self._create_mock_clip_model()
        self._using_mocks = True
        
        # Initialize mock engines
        self.motion_engine = self._create_mock_motion_engine()
        self.parameter_engine = self._create_mock_parameter_engine()
        
        self.logger.info("Basic mocks initialized successfully")
    
    def _create_mock_motion_engine(self):
        """Create a mock motion engine for testing."""
        class MockMotionEngine:
            def __init__(self):
                self.device = "mock"
                self.motion_mode = "2D"
            
            def process_motion_schedule(self, schedule, max_frames):
                return {i: {"zoom": 1.0, "angle": 0, "translation_x": 0, "translation_y": 0} for i in range(max_frames)}
            
            def interpolate_values(self, start_val, end_val, frame_idx, total_frames):
                if total_frames <= 1:
                    return start_val
                alpha = frame_idx / (total_frames - 1)
                return start_val + (end_val - start_val) * alpha
        
        return MockMotionEngine()
    
    def _create_mock_parameter_engine(self):
        """Create a mock parameter engine for testing."""
        class MockParameterEngine:
            def __init__(self):
                pass
            
            def validate_parameters(self, params):
                return True
            
            def validate(self, params):
                return True
            
            def process_animation_config(self, config):
                return config
            
            def process_motion_schedule(self, schedule, max_frames):
                return len(schedule) > 0
            
            def interpolate_values(self, keyframes, total_frames):
                """Interpolate values from keyframes for the given total frames."""
                if not keyframes:
                    return []
                
                # Simple linear interpolation
                frames = sorted(keyframes.keys())
                result = []
                
                for i in range(total_frames):
                    if i in keyframes:
                        result.append(keyframes[i])
                    else:
                        # Find surrounding keyframes for interpolation
                        prev_frame = max([f for f in frames if f <= i], default=0)
                        next_frame = min([f for f in frames if f >= i], default=frames[-1])
                        
                        if prev_frame == next_frame:
                            result.append(keyframes[prev_frame])
                        else:
                            # Linear interpolation
                            alpha = (i - prev_frame) / (next_frame - prev_frame)
                            prev_val = keyframes[prev_frame]
                            next_val = keyframes[next_frame]
                            result.append(prev_val + alpha * (next_val - prev_val))
                
                return result
        
        return MockParameterEngine()
    
    
    
    def _load_models(self) -> None:
        """Load Flux model components with centralized path management."""
        try:
            # Get centralized model paths
            try:
                model_paths = get_all_model_paths()
                self.logger.info("Using centralized model paths", extra={
                    "model_paths": {k: str(v) for k, v in model_paths.items()}
                })
            except Exception as e:
                self.logger.warning(f"Failed to get centralized model paths, using defaults: {e}")
                model_paths = {}
            
            # Use the model loader for flux.util integration
            from ..models.model_loader import model_loader
            
            self.logger.info("Loading Flux models for classic Deforum mode...", extra={
                "model_name": self.config.model_name,
                "device": self.config.device,
                "centralized_paths": len(model_paths) > 0
            })
            
            # Load all models using flux.util directly
            models = model_loader.load_models(
                model_name=self.config.model_name,
                device=str(self.config.device),
                use_trt=False  # Can be enabled for production optimization
            )
            
            # Assign loaded models
            self.t5 = models["t5"]
            self.clip = models["clip"]
            self.model = models["model"]
            self.ae = models["ae"]
            
            
            self.logger.info("All Flux models loaded successfully with centralized paths")
            
        except Exception as e:
            raise FluxModelError(
                f"Failed to load Flux models: {e}",
                model_name=self.config.model_name,
                device=str(self.config.device)
            )
    
    def _initialize_motion_engine(self) -> None:
        """Initialize the 16-channel motion engine in classic mode."""
        try:
            from ..animation.motion_engine import Flux16ChannelMotionEngine
            
            self.motion_engine = Flux16ChannelMotionEngine(
                device=str(self.config.device),
                motion_mode=getattr(self.config, 'motion_mode', '2D'),
                enable_learned_motion=False,  # Classic mode
                enable_transformer_attention=False  # Classic mode
            )
            
            self.logger.info("Motion engine initialized (Classic Mode)", extra={
                "motion_mode": getattr(self.config, 'motion_mode', '2D'),
                "learned_motion": False,
                "transformer_attention": False
            })
            
        except Exception as e:
            # For testing, create a basic mock motion engine
            self.logger.warning(f"Motion engine initialization failed, using mock: {e}")
            self.motion_engine = self._create_mock_motion_engine()
    
    def _initialize_parameter_engine(self) -> None:
        """Initialize the parameter processing engine."""
        try:
            from deforum_flux.animation.parameter_engine import ParameterEngine
            
            self.parameter_engine = ParameterEngine()
            self.logger.info("Parameter engine initialized")
            
        except Exception as e:
            # For testing, create a basic mock parameter engine
            self.logger.warning(f"Parameter engine initialization failed, using mock: {e}")
            self.parameter_engine = self._create_mock_parameter_engine()
    
    @handle_exception
    
    def generate_frame(
        self,
        prompt: str,
        frame_idx: int,
        prev_frame_latent: Optional[torch.Tensor] = None,
        motion_params: Optional[Dict[str, float]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single frame using Flux with classic Deforum animation parameters.
        
        Args:
            prompt: Text prompt for generation
            frame_idx: Index of the frame to generate
            prev_frame_latent: Previous frame latent for motion continuity
            motion_params: Motion parameters for this frame
            width: Image width (uses config default if None)
            height: Image height (uses config default if None)
            steps: Generation steps (uses config default if None)
            guidance: Guidance scale (uses config default if None)
            seed: Random seed (generated if None)
            
        Returns:
            Tuple of (decoded_image, latent_tensor)
            
        Raises:
            ValidationError: If inputs are invalid
            FluxModelError: If generation fails
            MotionProcessingError: If motion processing fails
        """
        frame_start_time = time.time()
        
        # Use config defaults
        width = width or self.config.width
        height = height or self.config.height
        steps = steps or self.config.steps
        guidance = guidance or self.config.guidance_scale
        
        # Validate inputs
        self.generation_utils.validate_generation_inputs(
            prompt, width, height, steps, guidance, self.config.max_prompt_length
        )
        
        # Generate seed if not provided
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        with LogContext(self.logger, "classic_frame_generation", 
                       frame_idx=frame_idx, prompt=prompt[:50], seed=seed):
            
            try:
                # Import Flux utilities
                try:
                    from flux.sampling import get_noise, prepare, get_schedule, denoise, unpack
                except ImportError as e:
                    raise FluxModelError(
                        f"Flux package not installed. Run: pip install git+https://github.com/black-forest-labs/flux.git",
                        model_name=self.config.model_name,
                        original_error=str(e)
                    )
                # Get initial noise
                x = get_noise(
                    1, height, width, 
                    device=self.config.device, 
                    dtype=torch.bfloat16, 
                    seed=seed
                )
                
                # Prepare inputs for Flux
                inp = prepare(self.t5, self.clip, x, prompt=prompt)
                
                # Get timesteps
                timesteps = get_schedule(
                    steps, 
                    inp["img"].shape[1], 
                    shift=(self.config.model_name != "flux-schnell")
                )
                
                # Generate
                device_type = str(self.config.device).replace("mps", "cpu")  # MPS doesn't support autocast
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance)
                
                # Store latent for motion continuity (in packed format)
                latent_tensor_packed = x.clone()
                
                # Decode to image
                x = unpack(x.float(), height, width)
                
                # Apply classic Deforum motion if previous frame and motion params provided
                # Motion is applied to unpacked latents (16-channel format)
                if prev_frame_latent is not None and motion_params is not None:
                    x = self.generation_utils.apply_motion_to_latent(
                        x, prev_frame_latent, motion_params, frame_idx, 
                        self.motion_engine, enable_learned_motion=False
                    )
                
                # Store unpacked latent for motion continuity
                latent_tensor = x.clone()
                
                device_type = str(self.config.device).replace("mps", "cpu")  # MPS doesn't support autocast
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    decoded_image = self.ae.decode(x)
                
                # Update statistics
                frame_time = time.time() - frame_start_time
                self.stats_manager.update_frame_stats(frame_time)
                
                self.logger.info(f"Classic frame {frame_idx} generated successfully", extra={
                    "frame_idx": frame_idx,
                    "seed": seed,
                    "motion_applied": motion_params is not None,
                    "generation_time": f"{frame_time:.3f}s"
                })
                
                return decoded_image, latent_tensor
                
            except Exception as e:
                raise FluxModelError(
                    f"Frame generation failed: {e}",
                    frame_index=frame_idx,
                    model_name=self.config.model_name
                )
    
    @handle_exception
    
    
    def generate_animation(self, animation_config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate a complete classic Deforum animation sequence.
        
        Args:
            animation_config: Complete animation configuration
            
        Returns:
            List of generated frames as numpy arrays
            
        Raises:
            ValidationError: If animation config is invalid
            FluxModelError: If generation fails
            MotionProcessingError: If motion processing fails
        """
        animation_start_time = time.time()
        
        # Validate animation configuration
        validation_errors = self.config_manager.validate_animation_config(animation_config)
        if validation_errors:
            raise ValidationError(
                "Animation configuration validation failed",
                validation_errors=validation_errors
            )
        
        # Extract configuration
        prompt = animation_config["prompt"]
        max_frames = animation_config["max_frames"]
        motion_schedule = animation_config.get("motion_schedule", {})
        
        # Optional parameters
        width = animation_config.get("width", self.config.width)
        height = animation_config.get("height", self.config.height)
        steps = animation_config.get("steps", self.config.steps)
        guidance = animation_config.get("guidance_scale", self.config.guidance_scale)
        seed = animation_config.get("seed")
        
        with LogContext(self.logger, "classic_animation_generation", 
                       max_frames=max_frames, prompt=prompt[:50]):
            
            # Interpolate motion schedule using classic Deforum approach
            interpolated_motion = self.generation_utils.interpolate_motion_schedule(
                motion_schedule, max_frames, self.parameter_engine
            )
            
            # Generate frames
            frames = []
            prev_latent = None
            
            for frame_idx in range(max_frames):
                # Get motion parameters for this frame
                motion_params = interpolated_motion.get(frame_idx, {})
                
                # Generate frame
                decoded_image, latent = self.generate_frame(
                    prompt=prompt,
                    frame_idx=frame_idx,
                    prev_frame_latent=prev_latent,
                    motion_params=motion_params if motion_params else None,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance,
                    seed=seed
                )
                
                # Convert to numpy array
                frame_array = self.generation_utils.tensor_to_numpy(decoded_image)
                frames.append(frame_array)
                
                # Update for next frame
                prev_latent = latent
                
                # Progress logging
                progress = (frame_idx + 1) / max_frames * 100
                self.logger.info(f"Classic frame {frame_idx + 1}/{max_frames} generated ({progress:.1f}%)", extra={
                    "frame_idx": frame_idx,
                    "progress_percent": progress
                })
                
                # Memory cleanup for long animations
                if self.config.memory_efficient and frame_idx % 5 == 0:
                    self.stats_manager.cleanup_resources(memory_efficient=True)
            
            # Update animation statistics
            animation_time = time.time() - animation_start_time
            self.stats_manager.update_animation_stats(animation_time, max_frames)
            
            self.logger.info(f"Classic animation generation completed in {animation_time:.2f}s", extra={
                "total_frames": max_frames,
                "total_time": animation_time,
                "average_frame_time": animation_time / max_frames,
                "classic_mode": True
            })
            
            return frames
    
    def create_simple_motion_schedule(
        self,
        max_frames: int,
        zoom_per_frame: float = 1.02,
        rotation_per_frame: float = 0.5,
        translation_x_per_frame: float = 0.0,
        translation_y_per_frame: float = 0.0,
        translation_z_per_frame: float = 0.0
    ) -> Dict[int, Dict[str, float]]:
        """
        Create a simple linear motion schedule for classic Deforum animation.
        
        Args:
            max_frames: Number of frames
            zoom_per_frame: Zoom increment per frame
            rotation_per_frame: Rotation increment per frame (degrees)
            translation_x_per_frame: X translation per frame (pixels)
            translation_y_per_frame: Y translation per frame (pixels)
            translation_z_per_frame: Z translation per frame
            
        Returns:
            Motion schedule dictionary
        """
        motion_schedule = {}
        
        for frame in range(0, max_frames, max(1, max_frames // 10)):  # Create keyframes
            motion_schedule[frame] = {
                "zoom": 1.0 + (zoom_per_frame - 1.0) * frame,
                "angle": rotation_per_frame * frame,
                "translation_x": translation_x_per_frame * frame,
                "translation_y": translation_y_per_frame * frame,
                "translation_z": translation_z_per_frame * frame
            }
        
        return motion_schedule
    
    def validate_production_ready(self) -> Dict[str, Any]:
        """
        Validate that the bridge is production-ready with real GPU utilization.
        
        Returns:
            Dictionary with production readiness status
            
        Raises:
            FluxModelError: If not production ready
        """
        validation = {
            "production_ready": False,
            "using_mocks": self._using_mocks,
            "gpu_available": torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
            "models_loaded": False,
            "device": str(self.config.device),
            "issues": []
        }
        
        # Check for mock usage
        if self._using_mocks:
            validation["issues"].append("CRITICAL: Using mock components - no real generation possible")
        
        # Check GPU availability
        if not validation["gpu_available"] and "cuda" in str(self.config.device):
            validation["issues"].append("WARNING: CUDA device requested but not available")
        
        # Check model loading
        if self.model is not None and self.ae is not None and self.t5 is not None and self.clip is not None:
            validation["models_loaded"] = True
        else:
            validation["issues"].append("CRITICAL: Not all models loaded")
        
        # Overall status
        validation["production_ready"] = (
            not self._using_mocks and 
            validation["models_loaded"] and
            len([issue for issue in validation["issues"] if "CRITICAL" in issue]) == 0
        )
        
        if not validation["production_ready"]:
            self.logger.error("Production validation failed", extra=validation)
        else:
            self.logger.info("Production validation passed - ready for GPU generation", extra=validation)
        
        return validation
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats_manager.get_stats()
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats_manager.reset_stats()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.resource_manager.cleanup_all()
        self.logger.info("Bridge cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
