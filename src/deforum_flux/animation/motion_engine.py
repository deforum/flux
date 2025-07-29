"""
Core Motion Engine for Classic Deforum 16-Channel Processing

This module contains the main Flux16ChannelMotionEngine class that orchestrates
classic Deforum-style motion processing for 16-channel Flux latents.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from .motion_transforms import MotionTransforms
from .motion_utils import MotionUtils
from deforum.core.exceptions import MotionProcessingError, TensorProcessingError
from deforum.core.logging_config import get_logger, log_performance, log_memory_usage
from deforum.utils.device_utils import normalize_device, get_torch_device, ensure_tensor_device


class Flux16ChannelMotionEngine(nn.Module):
    """
    Classic Deforum motion engine for 16-channel Flux latents.
    
    This engine focuses on geometric transformations and traditional parameter scheduling,
    providing the core functionality for classic Deforum-style animations.
    """
    
    def __init__(
        self, 
        config=None,  # Accept config parameter for compatibility
        device: str = "cpu",
        motion_mode: str = "grouped",  # "grouped", "independent", "mixed"
        enable_learned_motion: bool = False,  # Disabled for classic Deforum
        enable_transformer_attention: bool = False  # Disabled for classic Deforum
    ):
        """
        Initialize the classic 16-channel motion engine.
        
        Args:
            config: Configuration object (for compatibility with tests)
            device: Device to run on
            motion_mode: Motion processing mode (kept for compatibility)
            enable_learned_motion: Always False for classic mode
            enable_transformer_attention: Always False for classic mode
        """
        super().__init__()
        
        # Store config attribute for compatibility
        self.config = config
        
        # If config is provided, extract device from it
        if config is not None and hasattr(config, 'device'):
            device = config.device
        
        self.device = normalize_device(device)
        self.motion_mode = motion_mode
        self.enable_learned_motion = False  # Always disabled for classic mode
        self.enable_transformer_attention = False  # Always disabled for classic mode
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.motion_transforms = MotionTransforms(device=self.device)
        self.motion_utils = MotionUtils()
        
        # Move to device using torch device object
        self.to(get_torch_device(self.device))
        
        self.logger.info("Classic 16-Channel Flux Motion Engine initialized", extra={
            "motion_mode": motion_mode,
            "learned_motion": False,
            "transformer_attention": False,
            "classic_mode": True,
            "device": device,
            "config_provided": config is not None
        })
    
    @log_performance
    def apply_motion(
        self,
        flux_latent: torch.Tensor,
        motion_params: Dict[str, float],
        blend_factor: float = 1.0,
        use_learned_enhancement: bool = False,  # Always False for classic mode
        use_transformer_attention: bool = None,  # Always None/False for classic mode
        sequence_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply classic Deforum motion directly to 16-channel Flux latents.
        
        Args:
            flux_latent: Input Flux latent (B, 16, H, W) or sequence (B, T, 16, H, W)
            motion_params: Motion parameters (zoom, angle, translation_x, translation_y, translation_z)
            blend_factor: How much to blend motion (0=no motion, 1=full motion)
            use_learned_enhancement: Ignored (always False for classic mode)
            use_transformer_attention: Ignored (always False for classic mode)
            sequence_context: Ignored (not used in classic mode)
            
        Returns:
            Transformed Flux latent (same shape as input)
            
        Raises:
            TensorProcessingError: If tensor shapes are invalid
            MotionProcessingError: If motion processing fails
        """
        # Handle both single frame and sequence inputs
        is_sequence = len(flux_latent.shape) == 5
        
        if is_sequence:
            batch_size, seq_len, channels, height, width = flux_latent.shape
            if channels != 16:
                raise TensorProcessingError(
                    f"Expected 16-channel input, got {channels} channels",
                    tensor_shape=flux_latent.shape,
                    expected_shape=(batch_size, seq_len, 16, height, width)
                )
        else:
            if flux_latent.shape[1] != 16:
                raise TensorProcessingError(
                    f"Expected 16-channel input, got {flux_latent.shape[1]} channels",
                    tensor_shape=flux_latent.shape,
                    expected_shape=(flux_latent.shape[0], 16, flux_latent.shape[2], flux_latent.shape[3])
                )
            batch_size, channels, height, width = flux_latent.shape
            seq_len = 1
        
        try:
            # Always use classic motion processing (no transformer or learned components)
            if is_sequence:
                # Memory-optimized sequence processing (CRITICAL PERFORMANCE FIX)
                # Pre-allocate result tensor to avoid list accumulation and reduce memory usage
                result = torch.empty_like(flux_latent)
                
                # Process each frame in-place to minimize memory allocation
                for t in range(seq_len):
                    frame = flux_latent[:, t]  # (B, 16, H, W)
                    
                    # Apply motion directly to output tensor slice
                    result[:, t] = self._apply_classic_motion(
                        frame, motion_params, blend_factor
                    )
                    
                    # Periodic garbage collection for long sequences
                    if t % 10 == 0 and t > 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                result = self._apply_classic_motion(
                    flux_latent, motion_params, blend_factor
                )
            
            return result
            
        except Exception as e:
            raise MotionProcessingError(
                f"Classic motion application failed: {e}",
                motion_params=motion_params
            )
    
    def _apply_classic_motion(
        self,
        flux_latent: torch.Tensor,
        motion_params: Dict[str, float],
        blend_factor: float
    ) -> torch.Tensor:
        """Apply classic Deforum motion processing to a single frame."""
        # Validate input latent
        self.motion_utils.validate_latent(flux_latent, self.device)
        
        # Apply geometric transformation (the core of classic Deforum)
        geometric_transformed = self.motion_transforms.apply_geometric_transform(
            flux_latent, motion_params
        )
        
        # No learned motion in classic mode - just geometric transforms
        enhanced = geometric_transformed
        
        # Blend with original based on blend_factor
        if blend_factor < 1.0:
            result = flux_latent * (1 - blend_factor) + enhanced * blend_factor
        else:
            result = enhanced
        
        return result
    
    @log_performance
    @log_memory_usage
    def apply_motion_sequence(
        self,
        initial_latent: torch.Tensor,
        motion_sequence: List[Dict[str, float]],
        blend_factors: List[float]
    ) -> List[torch.Tensor]:
        """
        Apply a sequence of classic motion transformations.
        
        Args:
            initial_latent: Starting 16-channel latent
            motion_sequence: List of motion parameters for each frame
            blend_factors: List of blend factors for each frame
            
        Returns:
            List of transformed latents
            
        Raises:
            MotionProcessingError: If sequence processing fails
        """
        if len(motion_sequence) != len(blend_factors):
            raise MotionProcessingError(
                f"Motion sequence length ({len(motion_sequence)}) doesn't match "
                f"blend factors length ({len(blend_factors)})"
            )
        
        # Validate initial latent
        self.motion_utils.validate_latent(initial_latent, self.device)
        
        frames = [initial_latent]
        current_latent = initial_latent
        
        for i, (motion_params, blend_factor) in enumerate(zip(motion_sequence, blend_factors)):
            try:
                # Apply classic motion to current frame
                transformed = self.apply_motion(
                    current_latent,
                    motion_params,
                    blend_factor=blend_factor,
                    use_learned_enhancement=False  # Always disabled
                )
                
                frames.append(transformed)
                current_latent = transformed
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    self.logger.debug(f"Processed classic motion frame {i + 1}/{len(motion_sequence)}")
                
            except Exception as e:
                raise MotionProcessingError(
                    f"Failed to process classic motion frame {i}: {e}",
                    frame_index=i,
                    motion_params=motion_params
                )
        
        self.logger.info(f"Applied classic motion sequence to {len(frames)} frames")
        return frames
    
    def get_motion_statistics(self, latent: torch.Tensor) -> Dict[str, Any]:
        """
        Get statistical information about a latent tensor.
        
        Args:
            latent: 16-channel latent tensor
            
        Returns:
            Dictionary with statistical information
        """
        return self.motion_utils.get_motion_statistics(latent)
    
    def validate_latent(self, latent: torch.Tensor) -> None:
        """
        Validate that a latent tensor is suitable for processing.
        
        Args:
            latent: Latent tensor to validate
            
        Raises:
            TensorProcessingError: If validation fails
        """
        self.motion_utils.validate_latent(latent, self.device)
    
    def compare_latents(
        self, 
        latent1: torch.Tensor, 
        latent2: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compare two latent tensors to analyze motion effects.
        
        Args:
            latent1: First latent tensor (e.g., original)
            latent2: Second latent tensor (e.g., after motion)
            
        Returns:
            Dictionary with comparison metrics
        """
        return self.motion_utils.compare_latents(latent1, latent2)
    
    def create_motion_mask(
        self, 
        latent: torch.Tensor, 
        motion_type: str = "uniform"
    ) -> torch.Tensor:
        """
        Create a motion mask for selective motion application.
        
        Args:
            latent: Input latent tensor
            motion_type: Type of motion mask
            
        Returns:
            Motion mask tensor
        """
        return self.motion_utils.create_motion_mask(latent, motion_type)
    
    def interpolate_latents(
        self, 
        latent1: torch.Tensor, 
        latent2: torch.Tensor, 
        num_steps: int,
        interpolation_mode: str = "linear"
    ) -> List[torch.Tensor]:
        """
        Interpolate between two latents for smooth transitions.
        
        Args:
            latent1: Starting latent
            latent2: Ending latent
            num_steps: Number of interpolation steps
            interpolation_mode: Interpolation method
            
        Returns:
            List of interpolated latents
        """
        return self.motion_utils.interpolate_latents(
            latent1, latent2, num_steps, interpolation_mode
        )
    
    def optimize_motion_parameters(
        self,
        latent: torch.Tensor,
        target_motion: str = "smooth"
    ) -> Dict[str, float]:
        """
        Suggest optimal motion parameters based on latent characteristics.
        
        Args:
            latent: Input latent tensor
            target_motion: Type of desired motion
            
        Returns:
            Suggested motion parameters
        """
        return self.motion_utils.optimize_motion_parameters(latent, target_motion)
    
    def get_available_depth_models(self) -> Dict[str, bool]:
        """
        Check which depth models are available for Z-axis motion.
        
        Returns:
            Dictionary indicating model availability
        """
        return self.motion_transforms.get_available_depth_models()
    
    def apply_motion_with_mask(
        self,
        flux_latent: torch.Tensor,
        motion_params: Dict[str, float],
        motion_mask: Optional[torch.Tensor] = None,
        mask_type: str = "uniform",
        blend_factor: float = 1.0
    ) -> torch.Tensor:
        """
        Apply motion with selective masking for advanced effects.
        
        Args:
            flux_latent: Input latent tensor
            motion_params: Motion parameters
            motion_mask: Pre-computed motion mask (optional)
            mask_type: Type of mask to create if motion_mask is None
            blend_factor: Global blend factor
            
        Returns:
            Masked motion-transformed latent
        """
        # Create mask if not provided
        if motion_mask is None:
            motion_mask = self.create_motion_mask(flux_latent, mask_type)
        
        # Apply motion
        motion_result = self.apply_motion(flux_latent, motion_params, blend_factor=1.0)
        
        # Apply mask
        masked_result = flux_latent * (1 - motion_mask * blend_factor) + motion_result * (motion_mask * blend_factor)
        
        return masked_result
    
    def create_classic_zoom_sequence(
        self,
        initial_latent: torch.Tensor,
        num_frames: int,
        zoom_per_frame: float = 1.02,
        rotation_per_frame: float = 0.0,
        translation_per_frame: Dict[str, float] = None
    ) -> List[torch.Tensor]:
        """
        Create a classic Deforum-style zoom sequence.
        
        Args:
            initial_latent: Starting latent
            num_frames: Number of frames to generate
            zoom_per_frame: Zoom increment per frame
            rotation_per_frame: Rotation increment per frame (degrees)
            translation_per_frame: Translation increments per frame
            
        Returns:
            List of transformed latents creating zoom sequence
        """
        if translation_per_frame is None:
            translation_per_frame = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        motion_sequence = []
        blend_factors = []
        
        for frame in range(num_frames):
            motion_params = {
                "zoom": zoom_per_frame,
                "angle": rotation_per_frame,
                "translation_x": translation_per_frame.get("x", 0.0),
                "translation_y": translation_per_frame.get("y", 0.0),
                "translation_z": translation_per_frame.get("z", 0.0)
            }
            motion_sequence.append(motion_params)
            blend_factors.append(1.0)  # Full motion blend
        
        return self.apply_motion_sequence(initial_latent, motion_sequence, blend_factors)
    
    def create_orbital_motion_sequence(
        self,
        initial_latent: torch.Tensor,
        num_frames: int,
        orbit_radius: float = 20.0,
        orbit_speed: float = 2.0,
        zoom_factor: float = 1.01
    ) -> List[torch.Tensor]:
        """
        Create an orbital motion sequence (combination of rotation and translation).
        
        Args:
            initial_latent: Starting latent
            num_frames: Number of frames
            orbit_radius: Radius of orbital motion (pixels)
            orbit_speed: Speed of orbit (degrees per frame)
            zoom_factor: Zoom factor per frame
            
        Returns:
            List of transformed latents creating orbital motion
        """
        import math
        
        motion_sequence = []
        blend_factors = []
        
        for frame in range(num_frames):
            angle_rad = math.radians(frame * orbit_speed)
            
            motion_params = {
                "zoom": zoom_factor,
                "angle": frame * orbit_speed * 0.1,  # Slight rotation
                "translation_x": orbit_radius * math.cos(angle_rad),
                "translation_y": orbit_radius * math.sin(angle_rad),
                "translation_z": math.sin(angle_rad * 2) * 10.0  # Depth variation
            }
            motion_sequence.append(motion_params)
            blend_factors.append(1.0)
        
        return self.apply_motion_sequence(initial_latent, motion_sequence, blend_factors)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the motion engine configuration.
        
        Returns:
            Dictionary with engine information
        """
        return {
            "engine_type": "Flux16ChannelMotionEngine",
            "mode": "classic_deforum",
            "device": str(self.device),
            "motion_mode": self.motion_mode,
            "learned_motion_enabled": self.enable_learned_motion,
            "transformer_attention_enabled": self.enable_transformer_attention,
            "available_depth_models": self.get_available_depth_models(),
            "supported_motion_params": [
                "zoom", "angle", "translation_x", "translation_y", "translation_z"
            ],
            "supported_interpolation_modes": [
                "linear", "cubic", "slerp"
            ],
            "supported_mask_types": [
                "uniform", "center", "edges", "gradient"
            ]
        }



__all__ = ["Flux16ChannelMotionEngine"]
