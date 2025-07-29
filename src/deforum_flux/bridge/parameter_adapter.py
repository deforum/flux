"""
Parameter adapter for Flux-Deforum integration

This module provides parameter conversion and adaptation utilities
for bridging Deforum animation parameters with Flux generation.
"""

import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List

from deforum.core.exceptions import ParameterError
from deforum.core.logging_config import get_logger


class FluxDeforumParameterAdapter:
    """Adapter to convert Deforum parameters to Flux-compatible format."""
    
    def __init__(self):
        """Initialize the parameter adapter."""
        self.logger = get_logger(__name__)
    
    @staticmethod
    def adapt_strength_to_flux_timesteps(deforum_strength: float, max_steps: int = 20) -> Tuple[int, int]:
        """
        Convert Deforum strength (0.0-1.0) to Flux timestep range.
        
        Args:
            deforum_strength: Deforum strength value
            max_steps: Maximum number of sampling steps
            
        Returns:
            Tuple of (start_timestep, end_timestep)
        """
        # Higher strength = more denoising = start from higher noise level
        start_step = int((1.0 - deforum_strength) * max_steps)
        end_step = max_steps
        return start_step, end_step
    
    @staticmethod
    def prepare_flux_inputs(
        prompt: str, 
        width: int, 
        height: int, 
        init_image: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Prepare inputs for Flux generation in a format compatible with Deforum workflows.
        
        Args:
            prompt: Text prompt
            width: Image width
            height: Image height
            init_image: Optional initial image tensor
            
        Returns:
            Dictionary of prepared inputs
        """
        return {
            "prompt": prompt,
            "width": width,
            "height": height,
            "init_image": init_image
        }
    
    def convert_deforum_motion_to_cv2_matrix(
        self, 
        motion_params: Dict[str, float], 
        width: int = 512, 
        height: int = 512
    ) -> np.ndarray:
        """
        Convert Deforum motion parameters to OpenCV transformation matrix.
        
        Args:
            motion_params: Dictionary with motion parameters
            width: Image width
            height: Image height
            
        Returns:
            3x3 transformation matrix
        """
        # Extract motion parameters with defaults
        zoom = motion_params.get("zoom", 1.0)
        angle = motion_params.get("angle", 0.0)
        translation_x = motion_params.get("translation_x", 0.0)
        translation_y = motion_params.get("translation_y", 0.0)
        
        # Center point for rotation and scaling
        center_x, center_y = width / 2, height / 2
        
        # Create rotation and scaling matrix
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, zoom)
        
        # Add translation
        rotation_matrix[0, 2] += translation_x
        rotation_matrix[1, 2] += translation_y
        
        # Convert to 3x3 matrix
        transformation_matrix = np.eye(3)
        transformation_matrix[:2, :] = rotation_matrix
        
        return transformation_matrix
    
    def apply_motion_to_image(
        self, 
        image: np.ndarray, 
        motion_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply motion transformation to an image using OpenCV.
        
        Args:
            image: Input image as numpy array
            motion_params: Motion parameters
            
        Returns:
            Transformed image
        """
        height, width = image.shape[:2]
        
        # Get transformation matrix
        matrix = self.convert_deforum_motion_to_cv2_matrix(motion_params, width, height)
        
        # Apply transformation
        transformed = cv2.warpAffine(
            image, 
            matrix[:2, :], 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return transformed
    
    def convert_deforum_prompts_to_flux_schedule(
        self, 
        deforum_prompts: Dict[str, str], 
        max_frames: int
    ) -> List[str]:
        """
        Convert Deforum prompt schedule to frame-by-frame list for Flux.
        
        Args:
            deforum_prompts: Dictionary mapping frame numbers to prompts
            max_frames: Total number of frames
            
        Returns:
            List of prompts for each frame
        """
        prompts = []
        
        # Sort prompt keyframes
        sorted_prompts = sorted([(int(k), v) for k, v in deforum_prompts.items()])
        
        for frame_idx in range(max_frames):
            # Find the active prompt for this frame
            active_prompt = None
            for frame_num, prompt in sorted_prompts:
                if frame_num <= frame_idx:
                    active_prompt = prompt
                else:
                    break
            
            # Use the active prompt or default
            if active_prompt is None and sorted_prompts:
                active_prompt = sorted_prompts[0][1]
            elif active_prompt is None:
                active_prompt = "a beautiful landscape"
            
            prompts.append(active_prompt)
        
        return prompts
    
    def validate_motion_parameters(self, motion_params: Dict[str, float]) -> None:
        """
        Validate motion parameters for safety and compatibility.
        
        Args:
            motion_params: Motion parameters to validate
            
        Raises:
            ParameterError: If parameters are invalid
        """
        # Define safe ranges
        safe_ranges = {
            "zoom": (0.1, 10.0),
            "angle": (-360.0, 360.0),
            "translation_x": (-2000.0, 2000.0),
            "translation_y": (-2000.0, 2000.0),
            "translation_z": (-2000.0, 2000.0),
            "rotation_3d_x": (-360.0, 360.0),
            "rotation_3d_y": (-360.0, 360.0),
            "rotation_3d_z": (-360.0, 360.0)
        }
        
        for param_name, param_value in motion_params.items():
            if param_name in safe_ranges:
                min_val, max_val = safe_ranges[param_name]
                if not (min_val <= param_value <= max_val):
                    raise ParameterError(
                        f"Motion parameter {param_name} out of safe range [{min_val}, {max_val}]: {param_value}",
                        parameter_name=param_name,
                        parameter_value=param_value
                    )
    
    def interpolate_motion_parameters(
        self, 
        keyframes: Dict[int, Dict[str, float]], 
        total_frames: int
    ) -> List[Dict[str, float]]:
        """
        Interpolate motion parameters between keyframes.
        
        Args:
            keyframes: Dictionary mapping frame numbers to motion parameters
            total_frames: Total number of frames
            
        Returns:
            List of motion parameters for each frame
        """
        if not keyframes:
            return [{}] * total_frames
        
        # Get all parameter names
        all_params = set()
        for frame_params in keyframes.values():
            all_params.update(frame_params.keys())
        
        # Interpolate each parameter
        interpolated_frames = []
        
        for frame_idx in range(total_frames):
            frame_params = {}
            
            for param_name in all_params:
                # Find surrounding keyframes
                before_frame = None
                after_frame = None
                
                for kf_frame in sorted(keyframes.keys()):
                    if kf_frame <= frame_idx and param_name in keyframes[kf_frame]:
                        before_frame = kf_frame
                    elif kf_frame > frame_idx and param_name in keyframes[kf_frame] and after_frame is None:
                        after_frame = kf_frame
                        break
                
                # Interpolate value
                if before_frame is None and after_frame is not None:
                    # Before first keyframe
                    frame_params[param_name] = keyframes[after_frame][param_name]
                elif before_frame is not None and after_frame is None:
                    # After last keyframe
                    frame_params[param_name] = keyframes[before_frame][param_name]
                elif before_frame is not None and after_frame is not None:
                    # Between keyframes - linear interpolation
                    before_value = keyframes[before_frame][param_name]
                    after_value = keyframes[after_frame][param_name]
                    
                    if before_frame == after_frame:
                        frame_params[param_name] = before_value
                    else:
                        t = (frame_idx - before_frame) / (after_frame - before_frame)
                        interpolated_value = before_value + t * (after_value - before_value)
                        frame_params[param_name] = interpolated_value
                else:
                    # No keyframes for this parameter
                    frame_params[param_name] = 0.0
            
            interpolated_frames.append(frame_params)
        
        return interpolated_frames
    
    def convert_strength_schedule_to_flux(
        self, 
        strength_schedule: str, 
        max_frames: int, 
        max_steps: int = 20
    ) -> List[Tuple[int, int]]:
        """
        Convert Deforum strength schedule to Flux timestep ranges.
        
        Args:
            strength_schedule: Deforum strength schedule string
            max_frames: Total number of frames
            max_steps: Maximum sampling steps
            
        Returns:
            List of (start_step, end_step) tuples for each frame
        """
        from deforum.animation.parameter_engine import ParameterEngine
        
        # Parse strength schedule
        param_engine = ParameterEngine()
        strength_keyframes = param_engine.parse_keyframe_string(strength_schedule)
        strength_values = param_engine.interpolate_values(strength_keyframes, max_frames)
        
        # Convert to timestep ranges
        timestep_ranges = []
        for strength in strength_values:
            start_step, end_step = self.adapt_strength_to_flux_timesteps(strength, max_steps)
            timestep_ranges.append((start_step, end_step))
        
        return timestep_ranges
    
    def create_flux_compatible_config(
        self, 
        deforum_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create Flux-compatible configuration from Deforum parameters.
        
        Args:
            deforum_config: Deforum configuration dictionary
            
        Returns:
            Flux-compatible configuration
        """
        flux_config = {
            "width": deforum_config.get("width", 512),
            "height": deforum_config.get("height", 512),
            "num_inference_steps": deforum_config.get("steps", 20),
            "guidance_scale": deforum_config.get("guidance_scale", 7.5),
            "max_frames": deforum_config.get("max_frames", 30)
        }
        
        # Convert prompts if present
        if "prompts" in deforum_config:
            flux_config["prompt_schedule"] = self.convert_deforum_prompts_to_flux_schedule(
                deforum_config["prompts"], 
                flux_config["max_frames"]
            )
        
        # Convert motion parameters if present
        if "motion_schedule" in deforum_config:
            flux_config["motion_frames"] = self.interpolate_motion_parameters(
                deforum_config["motion_schedule"],
                flux_config["max_frames"]
            )
        
        # Convert strength schedule if present
        if "strength_schedule" in deforum_config:
            flux_config["timestep_ranges"] = self.convert_strength_schedule_to_flux(
                deforum_config["strength_schedule"],
                flux_config["max_frames"],
                flux_config["num_inference_steps"]
            )
        
        return flux_config
    
    def log_parameter_conversion(
        self, 
        original_params: Dict[str, Any], 
        converted_params: Dict[str, Any]
    ) -> None:
        """
        Log parameter conversion for debugging.
        
        Args:
            original_params: Original Deforum parameters
            converted_params: Converted Flux parameters
        """
        self.logger.debug("Parameter conversion completed", extra={
            "original_param_count": len(original_params),
            "converted_param_count": len(converted_params),
            "conversion_type": "deforum_to_flux"
        })
        
        # Log specific conversions
        both_params = original_params.keys() & converted_params.keys()
        for key in ["width", "height", "max_frames"]:
            if key in both_params:
                if original_params[key] != converted_params[key]:
                    self.logger.debug(f"Parameter {key} converted: {original_params[key]} -> {converted_params[key]}")
    
    def get_default_motion_params(self) -> Dict[str, float]:
        """
        Get default motion parameters.
        
        Returns:
            Dictionary with default motion parameters
        """
        return {
            "zoom": 1.0,
            "angle": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0,
            "rotation_3d_x": 0.0,
            "rotation_3d_y": 0.0,
            "rotation_3d_z": 0.0
        }
    
    def blend_motion_params(
        self, 
        params1: Dict[str, float], 
        params2: Dict[str, float], 
        blend_factor: float
    ) -> Dict[str, float]:
        """
        Blend two sets of motion parameters.
        
        Args:
            params1: First set of motion parameters
            params2: Second set of motion parameters
            blend_factor: Blending factor (0.0 = params1, 1.0 = params2)
            
        Returns:
            Blended motion parameters
        """
        blended = {}
        
        all_keys = set(params1.keys()) | set(params2.keys())
        
        for key in all_keys:
            val1 = params1.get(key, 0.0)
            val2 = params2.get(key, 0.0)
            blended[key] = val1 * (1 - blend_factor) + val2 * blend_factor
        
        return blended 