"""
Parameter processing engine for Deforum Flux

This module handles parameter parsing, interpolation, and validation,
consolidating the scattered parameter processing identified in the audit.
"""

import re
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

from deforum.core.exceptions import ParameterError, ValidationError
from deforum.core.logging_config import get_logger
from deforum.config.validation_utils import DomainValidators, ValidationUtils
from deforum.config.validation_rules import ValidationRules


class ParameterEngine:
    """
    Engine for processing Deforum animation parameters.
    
    Handles keyframe parsing, interpolation, and parameter validation
    with proper error handling and logging.
    """
    
    def __init__(self, config=None):
        """
        Initialize the parameter engine.
        
        Args:
            config: Optional configuration object for compatibility with tests and other components
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.logger.info("Parameter engine initialized", extra={
            "config_provided": config is not None
        })
    
    def parse_keyframe_string(self, keyframe_string: str) -> Dict[int, float]:
        """
        Parse keyframe string into frame->value mapping.
        
        Args:
            keyframe_string: String like "0:(1.0), 30:(1.5), 60:(1.0)"
            
        Returns:
            Dictionary mapping frame numbers to values
            
        Raises:
            ParameterError: If parsing fails
        """
        if not isinstance(keyframe_string, str):
            raise ParameterError(
                "Keyframe string must be a string",
                parameter_value=keyframe_string
            )
        
        keyframes = {}
        
        try:
            # Split by comma and parse each keyframe
            parts = keyframe_string.split(",")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                if ":" not in part:
                    self.logger.warning(f"Skipping invalid keyframe part: {part}")
                    continue
                    
                try:
                    frame_part, value_part = part.split(":", 1)
                    frame_num = int(frame_part.strip())
                    
                    # Validate frame number
                    frame_errors = ValidationUtils.validate_frame_number(frame_num)
                    if frame_errors:
                        self.logger.warning(f"Invalid frame number in '{part}': {frame_errors}")
                        continue
                    
                    # Extract value from parentheses
                    value_match = re.search(r'\((.*?)\)', value_part)
                    if value_match:
                        value = float(value_match.group(1))
                        keyframes[frame_num] = value
                    else:
                        self.logger.warning(f"No parentheses found in value part: {value_part}")
                        
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse keyframe part '{part}': {e}")
                    continue
            
            if not keyframes:
                raise ParameterError(
                    f"No valid keyframes found in string: {keyframe_string}",
                    parameter_value=keyframe_string
                )
            
            self.logger.debug(f"Parsed {len(keyframes)} keyframes from: {keyframe_string}")
            return keyframes
            
        except Exception as e:
            raise ParameterError(
                f"Failed to parse keyframe string: {e}",
                parameter_value=keyframe_string
            )
    
    def interpolate_values(self, keyframes: Dict[int, float], total_frames: int) -> List[float]:
        """
        Interpolate values between keyframes for all frames.
        
        Args:
            keyframes: Dictionary mapping frame numbers to values
            total_frames: Total number of frames to generate
            
        Returns:
            List of interpolated values for each frame
            
        Raises:
            ParameterError: If interpolation fails
        """
        if not keyframes:
            self.logger.warning("No keyframes provided, returning zeros")
            return [0.0] * total_frames
        
        if total_frames <= 0:
            raise ParameterError(
                f"Total frames must be positive, got {total_frames}",
                parameter_value=total_frames
            )
        
        try:
            # Sort keyframes by frame number
            sorted_keyframes = sorted(keyframes.items())
            
            values = []
            for frame_idx in range(total_frames):
                # Find surrounding keyframes
                before_frame, before_value = None, None
                after_frame, after_value = None, None
                
                for kf_frame, kf_value in sorted_keyframes:
                    if kf_frame <= frame_idx:
                        before_frame, before_value = kf_frame, kf_value
                    elif kf_frame > frame_idx and after_frame is None:
                        after_frame, after_value = kf_frame, kf_value
                        break
                
                # Interpolate value
                if before_frame is None:
                    # Before first keyframe
                    values.append(sorted_keyframes[0][1])
                elif after_frame is None:
                    # After last keyframe
                    values.append(before_value)
                else:
                    # Between keyframes - linear interpolation
                    t = (frame_idx - before_frame) / (after_frame - before_frame)
                    interpolated_value = before_value + t * (after_value - before_value)
                    values.append(interpolated_value)
            
            self.logger.debug(f"Interpolated {len(values)} values from {len(keyframes)} keyframes")
            return values
            
        except Exception as e:
            raise ParameterError(
                f"Failed to interpolate values: {e}",
                keyframes=keyframes,
                total_frames=total_frames
            )
    
    def parse_motion_schedule(self, motion_config: Dict[str, str]) -> Dict[int, Dict[str, float]]:
        """
        Parse motion configuration into a complete motion schedule.
        
        Args:
            motion_config: Dictionary with parameter names as keys and keyframe strings as values
            
        Returns:
            Dictionary mapping frame numbers to motion parameters
            
        Raises:
            ParameterError: If parsing fails
            ValidationError: If motion parameter names are invalid
        """
        motion_schedule = {}
        
        try:
            # First, validate all motion parameter names upfront
            unknown_params = []
            for param_name in motion_config.keys():
                if param_name not in ValidationRules.MOTION_RANGES:
                    unknown_params.append(param_name)
            
            if unknown_params:
                raise ValidationError(
                    f"Unknown motion parameters: {unknown_params}. "
                    f"Valid parameters: {list(ValidationRules.MOTION_RANGES.keys())}"
                )
            
            # Parse each parameter
            all_keyframes = {}
            for param_name, keyframe_string in motion_config.items():
                try:
                    keyframes = self.parse_keyframe_string(keyframe_string)
                    
                    # Validate parameter values at keyframes
                    min_val, max_val = ValidationRules.get_motion_range(param_name)
                    for frame, value in keyframes.items():
                        value_errors = ValidationUtils.validate_range(
                            value, min_val, max_val, f"{param_name}[frame_{frame}]", (int, float)
                        )
                        if value_errors:
                            raise ValidationError(
                                f"Invalid value for {param_name} at frame {frame}",
                                validation_errors=value_errors
                            )
                    
                    all_keyframes[param_name] = keyframes
                    self.logger.debug(f"Validated and parsed {param_name}: {len(keyframes)} keyframes")
                    
                except ParameterError as e:
                    self.logger.error(f"Failed to parse {param_name}: {e}")
                    raise ParameterError(
                        f"Failed to parse motion parameter {param_name}",
                        parameter_name=param_name,
                        parameter_value=keyframe_string
                    )
                except ValidationError as e:
                    self.logger.error(f"Validation failed for {param_name}: {e}")
                    raise
            
            # Find all unique frame numbers
            all_frames = set()
            for keyframes in all_keyframes.values():
                all_frames.update(keyframes.keys())
            
            # Build motion schedule
            for frame in sorted(all_frames):
                motion_schedule[frame] = {}
                for param_name, keyframes in all_keyframes.items():
                    # Use the exact value if available, or interpolate
                    if frame in keyframes:
                        motion_schedule[frame][param_name] = keyframes[frame]
                    else:
                        # Find surrounding keyframes for interpolation
                        before_frame = None
                        after_frame = None
                        
                        for kf_frame in sorted(keyframes.keys()):
                            if kf_frame < frame:
                                before_frame = kf_frame
                            elif kf_frame > frame and after_frame is None:
                                after_frame = kf_frame
                                break
                        
                        if before_frame is None:
                            # Before first keyframe
                            motion_schedule[frame][param_name] = keyframes[min(keyframes.keys())]
                        elif after_frame is None:
                            # After last keyframe
                            motion_schedule[frame][param_name] = keyframes[max(keyframes.keys())]
                        else:
                            # Interpolate
                            t = (frame - before_frame) / (after_frame - before_frame)
                            before_value = keyframes[before_frame]
                            after_value = keyframes[after_frame]
                            interpolated = before_value + t * (after_value - before_value)
                            motion_schedule[frame][param_name] = interpolated
            
            # Final validation of the complete schedule
            for frame, params in motion_schedule.items():
                self.validate_motion_parameters(params)
            
            self.logger.info(f"Created motion schedule with {len(motion_schedule)} keyframes")
            return motion_schedule
            
        except (ValidationError, ParameterError):
            # Re-raise validation and parameter errors
            raise
        except Exception as e:
            raise ParameterError(f"Failed to parse motion schedule: {e}")
    
    def validate_motion_parameters(self, motion_params: Dict[str, float]) -> None:
        """
        Validate motion parameters using centralized validation system.
        
        Args:
            motion_params: Dictionary of motion parameters to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Use centralized validation from DomainValidators
        errors = DomainValidators.validate_motion_params(motion_params)
        
        if errors:
            raise ValidationError(
                "Motion parameter validation failed",
                validation_errors=errors
            )
    
    def validate_motion_parameter_ranges(self, motion_params: Dict[str, float]) -> List[str]:
        """
        Validate motion parameter ranges using ValidationRules directly.
        
        Args:
            motion_params: Dictionary of motion parameters to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for param_name, param_value in motion_params.items():
            if param_name not in ValidationRules.MOTION_RANGES:
                errors.append(f"Unknown motion parameter: {param_name}")
                continue
            
            min_val, max_val = ValidationRules.get_motion_range(param_name)
            param_errors = ValidationUtils.validate_range(
                param_value, min_val, max_val, param_name, (int, float)
            )
            errors.extend(param_errors)
        
        return errors
    
    def smooth_motion_schedule(
        self, 
        motion_schedule: Dict[int, Dict[str, float]], 
        smoothing_factor: float = 0.1
    ) -> Dict[int, Dict[str, float]]:
        """
        Apply smoothing to motion schedule to reduce jitter.
        
        Args:
            motion_schedule: Original motion schedule
            smoothing_factor: Smoothing strength (0.0 = no smoothing, 1.0 = maximum smoothing)
            
        Returns:
            Smoothed motion schedule
            
        Raises:
            ValidationError: If smoothing_factor is invalid
        """
        # Validate smoothing factor
        smoothing_errors = ValidationUtils.validate_range(
            smoothing_factor, 0.0, 1.0, "smoothing_factor", (int, float)
        )
        if smoothing_errors:
            raise ValidationError(
                "Invalid smoothing factor",
                validation_errors=smoothing_errors
            )
        
        if not motion_schedule or smoothing_factor <= 0:
            return motion_schedule
        
        smoothed_schedule = {}
        sorted_frames = sorted(motion_schedule.keys())
        
        for i, frame in enumerate(sorted_frames):
            smoothed_schedule[frame] = {}
            original_params = motion_schedule[frame]
            
            for param_name, param_value in original_params.items():
                # Apply simple moving average smoothing
                smoothed_value = param_value
                
                if i > 0 and param_name in motion_schedule[sorted_frames[i-1]]:
                    prev_value = motion_schedule[sorted_frames[i-1]][param_name]
                    smoothed_value = (1 - smoothing_factor) * param_value + smoothing_factor * prev_value
                
                smoothed_schedule[frame][param_name] = smoothed_value
        
        self.logger.debug(f"Applied smoothing (factor={smoothing_factor}) to motion schedule")
        return smoothed_schedule
    
    def create_motion_schedule_from_deforum_config(self, deforum_config) -> Dict[int, Dict[str, float]]:
        """
        Create motion schedule from DeforumConfig object.
        
        Args:
            deforum_config: DeforumConfig object
            
        Returns:
            Motion schedule dictionary
            
        Raises:
            ValidationError: If motion parameters are invalid
        """
        # Build motion config with validation of parameter names
        motion_config = {}
        
        # Map config attributes to motion parameters (with validation)
        config_mappings = {
            "zoom": "zoom",
            "angle": "angle", 
            "translation_x": "translation_x",
            "translation_y": "translation_y",
            "translation_z": "translation_z",
            "rotation_3d_x": "rotation_3d_x",
            "rotation_3d_y": "rotation_3d_y",
            "rotation_3d_z": "rotation_3d_z"
        }
        
        for config_attr, motion_param in config_mappings.items():
            if self._safe_hasattr(deforum_config, config_attr):
                param_value = self._safe_getattr(deforum_config, config_attr)
                if param_value is not None:
                    motion_config[motion_param] = param_value
        
        if not motion_config:
            self.logger.warning("No motion parameters found in deforum_config")
            return {}
        
        return self.parse_motion_schedule(motion_config)
    
    def interpolate_schedule_to_frames(
        self, 
        schedule: Dict[int, Dict[str, float]], 
        total_frames: int
    ) -> List[Dict[str, float]]:
        """
        Interpolate a schedule to create values for every frame.
        
        Args:
            schedule: Schedule with keyframe values
            total_frames: Total number of frames needed
            
        Returns:
            List of parameter dictionaries, one per frame
            
        Raises:
            ParameterError: If total_frames is invalid
        """
        # Validate total_frames
        if total_frames <= 0:
            raise ParameterError(
                f"Total frames must be positive, got {total_frames}",
                parameter_value=total_frames
            )
        
        frame_params = []
        
        if not schedule:
            # Return empty parameters for all frames
            return [{}] * total_frames
        
        # Get all parameter names
        all_params = set()
        for frame_data in schedule.values():
            all_params.update(frame_data.keys())
        
        # Interpolate each parameter separately
        for frame_idx in range(total_frames):
            frame_data = {}
            
            for param_name in all_params:
                # Extract keyframes for this parameter
                param_keyframes = {}
                for frame, params in schedule.items():
                    if param_name in params:
                        param_keyframes[frame] = params[param_name]
                
                # Interpolate value for this frame
                if param_keyframes:
                    interpolated_values = self.interpolate_values(param_keyframes, total_frames)
                    frame_data[param_name] = interpolated_values[frame_idx]
                else:
                    frame_data[param_name] = 0.0
            
            frame_params.append(frame_data)
        
        return frame_params
    
    def parse_key_frames(self, keyframe_string: str) -> Dict[int, float]:
        """
        Compatibility method for parse_keyframe_string.
        
        Args:
            keyframe_string: String like "0:(1.0), 30:(1.5), 60:(1.0)"
            
        Returns:
            Dictionary mapping frame numbers to values
        """
        return self.parse_keyframe_string(keyframe_string)
    
    def get_inbetweens(self, keyframes: Dict[int, float], frame: int) -> float:
        """
        Get interpolated value for a specific frame from keyframes.
        
        Args:
            keyframes: Dictionary mapping frame numbers to values
            frame: Frame number to get value for
            
        Returns:
            Interpolated value for the specified frame
        """
        if not keyframes:
            return 0.0
        
        # Check if exact frame exists
        if frame in keyframes:
            return keyframes[frame]
        
        # Find surrounding keyframes for interpolation
        sorted_frames = sorted(keyframes.keys())
        
        # Before first keyframe
        if frame < sorted_frames[0]:
            return keyframes[sorted_frames[0]]
        
        # After last keyframe
        if frame > sorted_frames[-1]:
            return keyframes[sorted_frames[-1]]
        
        # Find surrounding keyframes
        before_frame = None
        after_frame = None
        
        for kf_frame in sorted_frames:
            if kf_frame <= frame:
                before_frame = kf_frame
            elif kf_frame > frame and after_frame is None:
                after_frame = kf_frame
                break
        
        if before_frame is None or after_frame is None:
            # Shouldn't happen but fallback
            return list(keyframes.values())[0]
        
        # Linear interpolation
        t = (frame - before_frame) / (after_frame - before_frame)
        before_value = keyframes[before_frame]
        after_value = keyframes[after_frame]
        
        return before_value + t * (after_value - before_value)

    def process_animation_config(self, animation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process animation configuration parameters for RunPod compatibility.
        
        This method provides a unified interface for processing animation configuration
        that may come from various sources (API, tests, presets).
        
        Args:
            animation_config: Dictionary containing animation parameters
            
        Returns:
            Processed and validated animation configuration
            
        Raises:
            ParameterError: If configuration processing fails
            ValidationError: If validation fails
        """
        try:
            self.logger.debug(f"Processing animation config with {len(animation_config)} parameters")
            
            processed_config = {}
            
            # Extract core animation parameters
            core_params = ["max_frames", "fps", "animation_mode", "width", "height"]
            for param in core_params:
                if param in animation_config:
                    processed_config[param] = animation_config[param]
            
            # Process motion parameters if present
            motion_params = {}
            motion_keys = ["zoom", "angle", "translation_x", "translation_y", "translation_z", 
                          "rotation_3d_x", "rotation_3d_y", "rotation_3d_z"]
            
            for key in motion_keys:
                if key in animation_config:
                    motion_params[key] = animation_config[key]
            
            # If we have motion parameters, process them into a motion schedule
            if motion_params:
                try:
                    motion_schedule = self.parse_motion_schedule(motion_params)
                    processed_config["motion_schedule"] = motion_schedule
                    self.logger.debug(f"Processed motion schedule with {len(motion_schedule)} keyframes")
                except Exception as e:
                    self.logger.warning(f"Failed to process motion schedule: {e}")
                    # Don't fail completely, just log the warning
            
            # Process strength schedules
            strength_params = ["strength_schedule", "noise_schedule", "contrast_schedule"]
            for param in strength_params:
                if param in animation_config:
                    if isinstance(animation_config[param], str):
                        try:
                            # Try to parse as keyframe string
                            parsed = self.parse_keyframe_string(animation_config[param])
                            processed_config[param] = parsed
                        except:
                            # If parsing fails, keep as string
                            processed_config[param] = animation_config[param]
                    else:
                        processed_config[param] = animation_config[param]
            
            # Copy through other parameters as-is
            other_keys = set(animation_config.keys()) - set(core_params) - set(motion_keys) - set(strength_params)
            for key in other_keys:
                processed_config[key] = animation_config[key]
            
            self.logger.info(f"Successfully processed animation config: {list(processed_config.keys())}")
            return processed_config
            
        except Exception as e:
            self.logger.error(f"Failed to process animation config: {e}")
            raise ParameterError(
                f"Animation config processing failed: {e}",
                parameter_value=animation_config
            )
    
    # SECURITY: Safe attribute access methods
    def _safe_hasattr(self, obj: Any, attr_name: str) -> bool:
        """
        Safely check if object has attribute with security validation.
        
        Args:
            obj: Object to check
            attr_name: Attribute name to check
            
        Returns:
            True if safe attribute exists, False otherwise
        """
        # Validate attribute name for security
        if not isinstance(attr_name, str):
            return False
        
        # Reject private/dunder attributes for security  
        if attr_name.startswith('_'):
            self.logger.warning(f"Rejecting access to private attribute: {attr_name}")
            return False
        
        # Reject potentially dangerous attributes
        dangerous_attrs = {
            '__class__', '__dict__', '__doc__', '__module__', '__weakref__',
            'exec', 'eval', 'compile', 'open', 'input', '__import__',
            'globals', 'locals', 'vars', 'dir', 'help'
        }
        
        if attr_name in dangerous_attrs:
            self.logger.warning(f"Rejecting access to dangerous attribute: {attr_name}")
            return False
        
        # Only allow alphanumeric and underscore in attribute names
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', attr_name):
            self.logger.warning(f"Rejecting invalid attribute name format: {attr_name}")
            return False
        
        # Check if attribute exists
        return hasattr(obj, attr_name)
    
    def _safe_getattr(self, obj: Any, attr_name: str, default: Any = None) -> Any:
        """
        Safely get attribute value with security validation.
        
        Args:
            obj: Object to get attribute from
            attr_name: Attribute name
            default: Default value if attribute doesn't exist
            
        Returns:
            Attribute value or default
            
        Raises:
            SecurityError: If attribute access is unsafe
        """
        # First check if we can safely access this attribute
        if not self._safe_hasattr(obj, attr_name):
            return default
        
        try:
            value = getattr(obj, attr_name, default)
            
            # Additional validation on the retrieved value
            if callable(value):
                self.logger.warning(f"Rejecting callable attribute: {attr_name}")
                return default
            
            return value
            
        except Exception as e:
            self.logger.warning(f"Error accessing attribute {attr_name}: {e}")
            return default

