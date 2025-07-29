"""
Configuration Management for Flux-Deforum Bridge

This module handles all configuration validation and management for the bridge.
Uses dependency injection to resolve circular dependencies.
"""

from typing import Dict, Any, List
from deforum.config.settings import Config
from deforum.core.exceptions import DeforumConfigError
from deforum.core.logging_config import get_logger

# Import dependency injection after core imports to avoid circular dependencies
try:
    from .dependency_config import get_dependency
    DEPENDENCY_INJECTION_AVAILABLE = True
except ImportError:
    DEPENDENCY_INJECTION_AVAILABLE = False


class BridgeConfigManager:
    """Manages configuration validation and settings for the Flux-Deforum bridge."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Use dependency injection if available, otherwise direct import
        if DEPENDENCY_INJECTION_AVAILABLE:
            try:
                self.validator = get_dependency('config_validator')
            except Exception as e:
                self.logger.warning(f"Could not get config validator via DI: {e}, falling back to direct import")
                self.validator = self._get_fallback_validator()
        else:
            self.validator = self._get_fallback_validator()
    
    def _get_fallback_validator(self):
        """Get fallback validator when dependency injection is not available."""
        try:
            from deforum.config.validation_utils import ValidationUtils
            return ValidationUtils()
        except ImportError:
            # Create a basic validator if ValidationUtils is not available
            return BasicValidator()
    
    def validate_config(self, config: Config) -> None:
        """
        Validate the configuration for the bridge.
        
        Args:
            config: Configuration object to validate
            
        Raises:
            DeforumConfigError: If configuration is invalid
        """
        try:
            if hasattr(self.validator, 'validate_config'):
                errors = self.validator.validate_config(config)
            else:
                errors = self._basic_validation(config)
                
            if errors:
                self.logger.error(f"Configuration validation failed: {errors}")
                raise DeforumConfigError(
                    "Configuration validation failed",
                    validation_errors=errors
                )
            
            # Additional bridge-specific validations
            self._validate_bridge_specific_config(config)
            
            self.logger.info("Configuration validation passed")
            
        except Exception as e:
            if isinstance(e, DeforumConfigError):
                raise
            else:
                self.logger.error(f"Unexpected error during validation: {e}")
                raise DeforumConfigError(f"Validation error: {e}")
    
    def _basic_validation(self, config: Config) -> List[str]:
        """Basic validation when ValidationUtils is not available."""
        errors = []
        
        # Basic dimension validation
        if not hasattr(config, 'width') or config.width <= 0:
            errors.append("Width must be positive")
        if not hasattr(config, 'height') or config.height <= 0:
            errors.append("Height must be positive")
        if not hasattr(config, 'steps') or config.steps <= 0:
            errors.append("Steps must be positive")
        if hasattr(config, 'guidance_scale') and config.guidance_scale <= 0:
            errors.append("Guidance scale must be positive")
            
        return errors
    
    def _validate_bridge_specific_config(self, config: Config) -> None:
        """Validate bridge-specific configuration requirements."""
        errors = []
        
        # Classic Deforum mode validations
        if hasattr(config, 'enable_learned_motion') and config.enable_learned_motion:
            self.logger.warning("Learned motion is enabled but classic Deforum mode is active")
        
        if hasattr(config, 'enable_transformer_attention') and config.enable_transformer_attention:
            self.logger.warning("Transformer attention is enabled but classic Deforum mode is active")
        
        # Motion mode validation
        valid_motion_modes = ["geometric", "learned", "hybrid"]
        if hasattr(config, 'motion_mode') and config.motion_mode not in valid_motion_modes:
            errors.append(f"Invalid motion_mode: {config.motion_mode}. Must be one of {valid_motion_modes}")
        
        # Device validation
        if not hasattr(config, 'device') or config.device is None:
            errors.append("Device must be specified")
        
        if errors:
            raise DeforumConfigError(
                "Bridge-specific configuration validation failed",
                validation_errors=errors
            )
    
    def validate_animation_config(self, animation_config: Dict[str, Any]) -> List[str]:
        """
        Validate animation configuration.
        
        Args:
            animation_config: Animation configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        try:
            if hasattr(self.validator, 'validate_animation_config'):
                return self.validator.validate_animation_config(animation_config)
            else:
                return self._basic_animation_validation(animation_config)
        except Exception as e:
            self.logger.error(f"Animation config validation error: {e}")
            return [f"Animation validation error: {e}"]
    
    def _basic_animation_validation(self, animation_config: Dict[str, Any]) -> List[str]:
        """Basic animation configuration validation."""
        errors = []
        
        if 'max_frames' in animation_config:
            if not isinstance(animation_config['max_frames'], int) or animation_config['max_frames'] <= 0:
                errors.append("max_frames must be a positive integer")
        
        if 'frame_rate' in animation_config:
            if not isinstance(animation_config['frame_rate'], (int, float)) or animation_config['frame_rate'] <= 0:
                errors.append("frame_rate must be a positive number")
        
        return errors
    
    def get_classic_deforum_defaults(self) -> Dict[str, Any]:
        """
        Get default configuration values for classic Deforum mode.
        
        Returns:
            Dictionary of default configuration values
        """
        return {
            "enable_learned_motion": False,
            "enable_transformer_attention": False,
            "motion_mode": "geometric",
            "memory_efficient": True,
            "skip_model_loading": False,
            "max_prompt_length": 512,
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance_scale": 3.5
        }
    
    def apply_classic_deforum_overrides(self, config: Config) -> Config:
        """
        Apply classic Deforum mode overrides to configuration.
        
        Args:
            config: Original configuration
            
        Returns:
            Configuration with classic Deforum overrides applied
        """
        # Ensure classic Deforum mode is enabled
        if hasattr(config, 'enable_learned_motion'):
            config.enable_learned_motion = False
        if hasattr(config, 'enable_transformer_attention'):
            config.enable_transformer_attention = False
        
        self.logger.info("Applied classic Deforum mode overrides to configuration")
        return config


class BasicValidator:
    """Basic validator fallback when advanced validation is not available."""
    
    def validate_config(self, config: Any) -> List[str]:
        """Basic configuration validation."""
        errors = []
        
        if not hasattr(config, 'width') or config.width <= 0:
            errors.append("Width must be positive")
        if not hasattr(config, 'height') or config.height <= 0:
            errors.append("Height must be positive")
        if not hasattr(config, 'steps') or config.steps <= 0:
            errors.append("Steps must be positive")
        
        return errors
    
    def validate_animation_config(self, animation_config: Dict[str, Any]) -> List[str]:
        """Basic animation configuration validation."""
        errors = []
        
        if 'max_frames' in animation_config:
            if not isinstance(animation_config['max_frames'], int) or animation_config['max_frames'] <= 0:
                errors.append("max_frames must be a positive integer")
        
        return errors
