"""
Generation Utilities for Flux-Deforum Bridge

This module contains utility functions for frame generation, motion application,
and parameter interpolation used by the bridge.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from deforum.core.exceptions import ValidationError, MotionProcessingError, TensorProcessingError
from deforum.core.logging_config import get_logger


class GenerationUtils:
    """Utility functions for generation and motion processing."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array with proper scaling.
        
        Args:
            tensor: Input tensor to convert
            
        Returns:
            Numpy array scaled to [0, 255] uint8 format
        """
        # Move to CPU and convert to float32
        array = tensor.cpu().float().numpy()
        
        # Handle batch dimension
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        
        # Transpose from CHW to HWC if needed
        if array.ndim == 3 and array.shape[0] in [1, 3, 4]:
            array = np.transpose(array, (1, 2, 0))
        
        # Clip and scale to [0, 255]
        array = np.clip(array, 0, 1)
        array = (array * 255).astype(np.uint8)
        
        return array
    
    def validate_generation_inputs(
        self, 
        prompt: str, 
        width: int, 
        height: int, 
        steps: int, 
        guidance: float,
        max_prompt_length: int = 512
    ) -> None:
        """
        Validate inputs for frame generation.
        
        Args:
            prompt: Text prompt
            width: Image width
            height: Image height  
            steps: Generation steps
            guidance: Guidance scale
            max_prompt_length: Maximum allowed prompt length
            
        Raises:
            ValidationError: If inputs are invalid
        """
        errors = []
        
        if not prompt or len(prompt.strip()) == 0:
            errors.append("Prompt cannot be empty")
        
        if len(prompt) > max_prompt_length:
            errors.append(f"Prompt too long: {len(prompt)} > {max_prompt_length}")
        
        if width < 64 or width > 4096:
            errors.append(f"Invalid width: {width} (must be 64-4096)")
        
        if height < 64 or height > 4096:
            errors.append(f"Invalid height: {height} (must be 64-4096)")
        
        if steps < 1 or steps > 200:
            errors.append(f"Invalid steps: {steps} (must be 1-200)")
        
        if guidance < 0.0 or guidance > 30.0:
            errors.append(f"Invalid guidance: {guidance} (must be 0.0-30.0)")
        
        if errors:
            raise ValidationError("Input validation failed", validation_errors=errors)
    
    def apply_motion_to_latent(
        self,
        current_latent: torch.Tensor,
        prev_frame_latent: torch.Tensor,
        motion_params: Dict[str, float],
        frame_idx: int,
        motion_engine,
        enable_learned_motion: bool = False
    ) -> torch.Tensor:
        """
        Apply motion transformation to latent tensor.
        
        Args:
            current_latent: Current frame latent
            prev_frame_latent: Previous frame latent
            motion_params: Motion parameters
            frame_idx: Frame index for error reporting
            motion_engine: Motion engine instance
            enable_learned_motion: Whether to use learned motion
            
        Returns:
            Motion-transformed latent tensor
            
        Raises:
            MotionProcessingError: If motion application fails
            TensorProcessingError: If tensor shapes are invalid
        """
        try:
            # Ensure we have 16 channels for motion processing
            if current_latent.shape[1] != 16:
                raise TensorProcessingError(
                    f"Expected 16-channel latent, got {current_latent.shape[1]} channels",
                    tensor_shape=current_latent.shape,
                    expected_shape=(current_latent.shape[0], 16, current_latent.shape[2], current_latent.shape[3])
                )
            
            # Apply motion using the motion engine
            motion_applied = motion_engine.apply_motion(
                prev_frame_latent,
                motion_params,
                blend_factor=0.3,  # Blend with current latent
                use_learned_enhancement=enable_learned_motion
            )
            
            # Blend with current latent for stability
            result = 0.7 * current_latent + 0.3 * motion_applied
            
            return result
            
        except Exception as e:
            raise MotionProcessingError(
                f"Motion application failed: {e}",
                frame_index=frame_idx,
                motion_params=motion_params
            )
    
    def interpolate_motion_schedule(
        self, 
        motion_schedule: Dict[int, Dict[str, float]], 
        total_frames: int,
        parameter_engine
    ) -> Dict[int, Dict[str, float]]:
        """
        Interpolate motion schedule for all frames using classic Deforum approach.
        
        Args:
            motion_schedule: Keyframe-based motion schedule
            total_frames: Total number of frames
            parameter_engine: Parameter engine for interpolation
            
        Returns:
            Interpolated motion parameters for each frame
        """
        if not motion_schedule:
            return {}
        
        interpolated = {}
        
        # Get all motion parameter names
        all_params = set()
        for frame_params in motion_schedule.values():
            all_params.update(frame_params.keys())
        
        self.logger.debug(f"Interpolating {len(all_params)} motion parameters across {total_frames} frames")
        
        # Interpolate each parameter using classic Deforum method
        for param_name in all_params:
            # Extract keyframes for this parameter
            keyframes = {}
            for frame, params in motion_schedule.items():
                if param_name in params:
                    keyframes[frame] = params[param_name]
            
            # Add default values at frame 0 and last frame if not present
            if 0 not in keyframes:
                # Use neutral defaults for classic Deforum parameters
                defaults = {
                    "zoom": 1.0,
                    "angle": 0.0,
                    "translation_x": 0.0,
                    "translation_y": 0.0,
                    "translation_z": 0.0
                }
                keyframes[0] = defaults.get(param_name, 0.0)
            
            if (total_frames - 1) not in keyframes and keyframes:
                # Extend last value to final frame
                last_frame = max(keyframes.keys())
                keyframes[total_frames - 1] = keyframes[last_frame]
            
            # Interpolate values using parameter engine
            if keyframes:
                interpolated_values = parameter_engine.interpolate_values(keyframes, total_frames)
                
                # Store in result
                for frame_idx, value in enumerate(interpolated_values):
                    if frame_idx not in interpolated:
                        interpolated[frame_idx] = {}
                    interpolated[frame_idx][param_name] = value
        
        self.logger.info(f"Motion schedule interpolated for {len(interpolated)} frames")
        return interpolated
    
    def create_classic_motion_schedule(
        self,
        max_frames: int,
        zoom_schedule: Optional[Dict[int, float]] = None,
        rotation_schedule: Optional[Dict[int, float]] = None,
        translation_x_schedule: Optional[Dict[int, float]] = None,
        translation_y_schedule: Optional[Dict[int, float]] = None,
        translation_z_schedule: Optional[Dict[int, float]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Create a classic Deforum-style motion schedule from individual parameter schedules.
        
        Args:
            max_frames: Maximum number of frames
            zoom_schedule: Zoom keyframes {frame: zoom_value}
            rotation_schedule: Rotation keyframes {frame: angle_degrees}
            translation_x_schedule: X translation keyframes {frame: x_pixels}
            translation_y_schedule: Y translation keyframes {frame: y_pixels}
            translation_z_schedule: Z translation keyframes {frame: z_value}
            
        Returns:
            Combined motion schedule
        """
        motion_schedule = {}
        
        # Combine all schedules
        schedules = {
            "zoom": zoom_schedule or {},
            "angle": rotation_schedule or {},
            "translation_x": translation_x_schedule or {},
            "translation_y": translation_y_schedule or {},
            "translation_z": translation_z_schedule or {}
        }
        
        # Get all keyframe indices
        all_frames = set()
        for schedule in schedules.values():
            all_frames.update(schedule.keys())
        
        # Build combined schedule
        for frame in all_frames:
            if frame < max_frames:
                motion_schedule[frame] = {}
                for param_name, schedule in schedules.items():
                    if frame in schedule:
                        motion_schedule[frame][param_name] = schedule[frame]
        
        return motion_schedule