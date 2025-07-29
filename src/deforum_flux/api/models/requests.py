"""
Request models for API endpoints
Updated to fully support all parameters from page.tsx DeforumFluxConfig interface
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator

class GenerationParameters(BaseModel):
    """Complete parameters matching page.tsx DeforumFluxConfig interface"""
    
    
    # Core Generation (from Generation tab)
    width: int = Field(default=1024, description="Image width (must be multiple of 64)", ge=512, le=2048)
    height: int = Field(default=1024, description="Image height (must be multiple of 64)", ge=512, le=2048)
    num_inference_steps: int = Field(default=20, description="Number of inference steps", ge=4, le=50)
    guidance_scale: float = Field(default=3.5, description="Guidance scale", ge=0.0, le=20.0)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    
    # Animation (from Animation tab)
    animation_mode: str = Field(default="2D", description="Animation mode")
    max_frames: int = Field(default=120, description="Maximum number of frames", ge=1, le=1000)
    
    # Motion Schedules (keyframe format: "frame:(value)")
    zoom: str = Field(default="0:(1.0)", description="Zoom schedule")
    angle: str = Field(default="0:(0)", description="Rotation angle schedule")
    translation_x: str = Field(default="0:(0)", description="X translation schedule")
    translation_y: str = Field(default="0:(0)", description="Y translation schedule")
    translation_z: str = Field(default="0:(0)", description="Z translation schedule")
    rotation_3d_x: str = Field(default="0:(0)", description="3D X rotation schedule")
    rotation_3d_y: str = Field(default="0:(0)", description="3D Y rotation schedule")
    rotation_3d_z: str = Field(default="0:(0)", description="3D Z rotation schedule")
    
    # Prompts (from Prompts tab)
    prompts: str = Field(default="0: A beautiful landscape, cinematic lighting, highly detailed", 
                        description="Primary prompt schedule", min_length=1, max_length=2000)
    prompt_2: str = Field(default="", description="Secondary prompt (T5 Encoder)")
    
    # Strength & Noise
    strength_schedule: str = Field(default="0:(0.75)", description="Denoising strength schedule")
    noise_schedule: str = Field(default="0:(0.02)", description="Noise schedule")
    
    # Memory & Performance (from Performance tab)
    enable_attention_slicing: bool = Field(default=False, description="Enable attention slicing")
    enable_vae_tiling: bool = Field(default=False, description="Enable VAE tiling")
    offload: bool = Field(default=True, description="Enable CPU offloading")
    
    # Output (from Output tab)
    output_type: str = Field(default="pil", description="Output tensor type")
    batch_name: str = Field(default="deforum_flux", description="Batch name for output")
    save_samples: bool = Field(default=True, description="Save intermediate samples")
    
    # Server Configuration (from Performance tab)
    api_port: int = Field(default=7860, description="API port", ge=1000, le=65535)
    api_host: str = Field(default="localhost", description="API host")
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")
        return v
    
    @validator('animation_mode')
    def validate_animation_mode(cls, v):
        allowed_modes = ["None", "2D", "3D", "Video Input"]
        if v not in allowed_modes:
            raise ValueError(f"Animation mode must be one of: {allowed_modes}")
        return v
    
    
    @validator('output_type')
    def validate_output_type(cls, v):
        allowed_types = ["pil", "np", "pt"]
        if v not in allowed_types:
            raise ValueError(f"Output type must be one of: {allowed_types}")
        return v

class AnimationConfig(BaseModel):
    """Animation configuration for complex sequences"""
    
    name: Optional[str] = Field(default=None, description="Animation name")
    description: Optional[str] = Field(default=None, description="Animation description")
    
    # Timing
    fps: int = Field(default=30, description="Frames per second", ge=1, le=60)
    duration: Optional[float] = Field(default=None, description="Duration in seconds", ge=0.1)
    
    # Effects
    enable_interpolation: bool = Field(default=True, description="Enable frame interpolation")
    interpolation_method: str = Field(default="linear", description="Interpolation method")
    
    # Post-processing
    apply_stabilization: bool = Field(default=False, description="Apply video stabilization")
    enhance_quality: bool = Field(default=False, description="Apply quality enhancement")
    
    @validator('interpolation_method')
    def validate_interpolation_method(cls, v):
        allowed_methods = ["linear", "cubic", "bezier"]
        if v not in allowed_methods:
            raise ValueError(f"Interpolation method must be one of: {allowed_methods}")
        return v

class GenerationRequest(BaseModel):
    """Complete generation request matching frontend interface"""
    
    parameters: GenerationParameters
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")
    animation_config: Optional[AnimationConfig] = Field(default=None)
    
    # Request metadata
    client_id: Optional[str] = Field(default=None, description="Client identifier")
    priority: int = Field(default=0, description="Request priority", ge=0, le=10)
    tags: List[str] = Field(default=[], description="Request tags for organization")
    
    # Callbacks
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for completion notification")
    
    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "prompts": "0: A beautiful sunset over mountains, cinematic lighting",
                    "width": 1024,
                    "height": 1024,
                    "max_frames": 30,
                    "num_inference_steps": 20,
                    "guidance_scale": 3.5,
                    "seed": -1,
                    "animation_mode": "2D",
                    "zoom": "0:(1.0), 15:(1.1), 30:(1.0)",
                    "angle": "0:(0), 15:(5), 30:(0)",
                    "translation_x": "0:(0)",
                    "translation_y": "0:(0)",
                    "translation_z": "0:(0)",
                    "rotation_3d_x": "0:(0)",
                    "rotation_3d_y": "0:(0)",
                    "rotation_3d_z": "0:(0)",
                    "prompt_2": "",
                    "strength_schedule": "0:(0.75)",
                    "noise_schedule": "0:(0.02)",
                    "enable_attention_slicing": False,
                    "enable_vae_tiling": False,
                    "offload": True,
                    "output_type": "pil",
                    "batch_name": "deforum_flux",
                    "save_samples": True,
                    "api_port": 7860,
                    "api_host": "localhost"
                },
                "animation_config": {
                    "name": "Sunset Animation",
                    "fps": 30,
                    "enable_interpolation": True
                }
            }
        }

# Simplified request for direct JSON payload from frontend
class DirectGenerationRequest(BaseModel):
    """Direct request format matching frontend JSON payload"""
    
    # Accept any parameters from frontend
    parameters: Dict[str, Any] = Field(..., description="Generation parameters from frontend")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")
    motion_schedules: Optional[Dict[str, str]] = Field(default=None, description="Motion schedules")
    
    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "prompt": "A beautiful sunset over mountains",
                    "width": 1024,
                    "height": 1024,
                    "max_frames": 30,
                    "steps": 20,
                    "guidance_scale": 3.5,
                    "seed": None,
                    "animation_mode": "2D",
                    "batch_name": "test_generation"
                },
                "config": {
                    "enable_attention_slicing": False,
                    "enable_vae_tiling": False,
                    "offload": True,
                    "output_type": "pil",
                    "save_samples": True
                },
                "motion_schedules": {
                    "zoom": "0:(1.0), 15:(1.1), 30:(1.0)",
                    "angle": "0:(0), 15:(5), 30:(0)",
                    "translation_x": "0:(0)",
                    "translation_y": "0:(0)",
                    "translation_z": "0:(0)",
                    "rotation_3d_x": "0:(0)",
                    "rotation_3d_y": "0:(0)",
                    "rotation_3d_z": "0:(0)"
                }
            }
        }

class PresetApplicationRequest(BaseModel):
    """Request to apply a preset to base parameters"""
    
    preset_id: str = Field(..., description="Preset identifier")
    base_parameters: Dict[str, Any] = Field(..., description="Base parameters to modify")
    override_existing: bool = Field(default=False, description="Override existing parameter values")
    
class ValidationRequest(BaseModel):
    """Request for parameter validation"""
    
    parameters: Dict[str, Any] = Field(..., description="Parameters to validate")
    strict_mode: bool = Field(default=False, description="Enable strict validation")
    
class JobActionRequest(BaseModel):
    """Request for job actions (cancel, pause, resume)"""
    
    action: str = Field(..., description="Action to perform")
    reason: Optional[str] = Field(default=None, description="Reason for action")
    
    @validator('action')
    def validate_action(cls, v):
        allowed_actions = ["cancel", "pause", "resume", "restart"]
        if v not in allowed_actions:
            raise ValueError(f"Action must be one of: {allowed_actions}")
        return v