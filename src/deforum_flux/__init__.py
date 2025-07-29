"""
Deforum Flux - Flux Model Integration and Animation Engine

This package provides Flux model integration for Deforum animations,
including the bridge, model management, animation engine, and generation API.
"""

try:
    from .models.models import ModelManager
except ImportError:
    # ModelManager might not be available in all environments
    ModelManager = None

try:
    from .bridge import FluxDeforumBridge
except ImportError:
    # FluxDeforumBridge might not be available in all environments
    FluxDeforumBridge = None

try:
    from .animation.motion_engine import Flux16ChannelMotionEngine as MotionEngine
    from .animation.parameter_engine import ParameterEngine
except ImportError:
    # Animation components might not be available in all environments
    MotionEngine = None
    ParameterEngine = None

__version__ = "0.1.0"
__all__ = []
if ModelManager:
    __all__.append("ModelManager")
if FluxDeforumBridge:
    __all__.append("FluxDeforumBridge")
if MotionEngine:
    __all__.append("MotionEngine")
if ParameterEngine:
    __all__.append("ParameterEngine")
