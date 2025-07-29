"""
Dependency Injection Configuration for Flux-Deforum Bridge

This module implements dependency injection to resolve circular dependencies
identified in the audit. It provides a centralized way to configure and
manage dependencies across the bridge components.
"""

from typing import Dict, Any, Optional, Type
from functools import lru_cache

from deforum.core.exceptions import DeforumConfigError, ValidationError
from deforum.core.logging_config import get_logger


class DependencyContainer:
    """Container for managing dependencies with lazy loading to prevent circular imports."""
    
    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        self.logger = get_logger(__name__)
    
    def register_factory(self, name: str, factory: callable) -> None:
        """Register a factory function for creating instances."""
        self._factories[name] = factory
        self.logger.debug(f"Registered factory for {name}")
    
    def register_instance(self, name: str, instance: Any) -> None:
        """Register a singleton instance."""
        self._instances[name] = instance
        self.logger.debug(f"Registered instance for {name}")
    
    @lru_cache(maxsize=128)
    def get(self, name: str) -> Any:
        """Get an instance by name, creating it if necessary."""
        if name in self._instances:
            return self._instances[name]
        
        if name in self._factories:
            try:
                instance = self._factories[name]()
                self._instances[name] = instance
                self.logger.debug(f"Created instance for {name}")
                return instance
            except Exception as e:
                self.logger.error(f"Failed to create instance for {name}: {e}")
                raise DeforumConfigError(f"Failed to create {name}", dependency_name=name)
        
        raise DeforumConfigError(f"No factory or instance registered for {name}", dependency_name=name)
    
    def clear_cache(self) -> None:
        """Clear the LRU cache and reset instances."""
        self.get.cache_clear()
        self._instances.clear()
        self.logger.debug("Cleared dependency cache")


# Global dependency container
_container = DependencyContainer()


def register_bridge_dependencies() -> None:
    """Register all bridge-related dependencies."""
    
    # Config validator factory
    def create_config_validator():
        """Factory for creating configuration validator."""
        try:
            from deforum.config.validation_utils import ValidationUtils
            return ValidationUtils()
        except ImportError:
            # Fallback validator if ValidationUtils is not available
            return BasicValidator()
    
    # Logger factory
    def create_bridge_logger():
        """Factory for creating bridge logger."""
        return get_logger("flux_deforum_bridge")
    
    # Exception handler factory
    def create_exception_handler():
        """Factory for creating exception handler."""
        from deforum.core.exceptions import handle_exception
        return handle_exception
    
    # Register factories
    _container.register_factory('config_validator', create_config_validator)
    _container.register_factory('bridge_logger', create_bridge_logger)
    _container.register_factory('exception_handler', create_exception_handler)


class BasicValidator:
    """Basic validator fallback when ValidationUtils is not available."""
    
    def validate_config(self, config: Any) -> list:
        """Basic configuration validation."""
        errors = []
        
        if not hasattr(config, 'width') or config.width <= 0:
            errors.append("Width must be positive")
        
        if not hasattr(config, 'height') or config.height <= 0:
            errors.append("Height must be positive")
        
        if not hasattr(config, 'steps') or config.steps <= 0:
            errors.append("Steps must be positive")
        
        return errors
    
    def validate_animation_config(self, animation_config: Dict[str, Any]) -> list:
        """Basic animation configuration validation."""
        errors = []
        
        if 'max_frames' in animation_config:
            if not isinstance(animation_config['max_frames'], int) or animation_config['max_frames'] <= 0:
                errors.append("max_frames must be a positive integer")
        
        return errors


def get_dependency(name: str) -> Any:
    """Get a dependency from the container."""
    return _container.get(name)


def configure_bridge_dependencies() -> None:
    """Configure all bridge dependencies."""
    try:
        register_bridge_dependencies()
        logger = get_logger(__name__)
        logger.info("Bridge dependencies configured successfully")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to configure bridge dependencies: {e}")
        raise DeforumConfigError("Failed to configure bridge dependencies", details={"error": str(e)})


def cleanup_dependencies() -> None:
    """Clean up dependency container."""
    _container.clear_cache()
