"""
Generation endpoints for video creation
Updated to support all parameters from page.tsx frontend interface
Fixed: Enhanced input sanitization and security validation
"""

import uuid
import asyncio
import re
from typing import Dict, Any, Optional, Union, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from deforum_flux.bridge import FluxDeforumBridge
from deforum.config.settings import Config
from deforum.core.exceptions import DeforumConfigError, FluxModelError, ValidationError
from deforum.core.logging_config import get_logger
from deforum_flux.api.models.requests import GenerationRequest, DirectGenerationRequest, AnimationConfig
from deforum_flux.api.models.responses import GenerationResponse, StatusResponse
# Simple job manager - replace with proper implementation
class JobManager:
    def __init__(self):
        self.jobs = {}
    def create_job(self, job_id, data):
        self.jobs[job_id] = {**data, "status": "created"}
    def update_job(self, job_id, updates):
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
    def get_job(self, job_id):
        return self.jobs.get(job_id)

router = APIRouter()
logger = get_logger(__name__)
job_manager = JobManager()

# Security constants
MAX_PROMPT_LENGTH = 2000
MAX_NUMERIC_VALUE = 1000000
ALLOWED_ANIMATION_MODES = ["2D", "3D", "Interpolation"]
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'javascript:',
    r'on\w+\s*=',
    r'eval\s*\(',
    r'exec\s*\(',
    r'import\s+',
    r'subprocess',
    r'os\.system',
    r'--',
    r';\s*(DROP|DELETE|INSERT|UPDATE|UNION|SELECT)',
    r'\.\./\.\./\.\.',  # Path traversal
    r'file://',
    r'ftp://',
    r'data:',
]

def is_malicious_input(value: str) -> bool:
    """Check if input contains malicious patterns"""
    if not isinstance(value, str):
        return False
    
    value_lower = value.lower()
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, value_lower, re.IGNORECASE):
            return True
    
    # Check for excessive length
    if len(value) > MAX_PROMPT_LENGTH:
        return True
    
    # Check for repeated characters (potential DoS)
    if len(set(value)) < len(value) * 0.1 and len(value) > 100:
        return True
    
    return False

def sanitize_string_input(value: Any, max_length: int = MAX_PROMPT_LENGTH) -> str:
    """Sanitize string input with comprehensive security checks"""
    if not isinstance(value, str):
        value = str(value)
    
    # Check for malicious patterns first
    if is_malicious_input(value):
        raise HTTPException(
            status_code=400, 
            detail="Input contains potentially malicious content"
        )
    
    # Remove HTML tags
    value = re.sub(r'<[^>]*>', '', value)
    
    # Remove SQL injection patterns
    value = re.sub(r'(--|;|DROP|DELETE|INSERT|UPDATE|UNION|SELECT)', '', value, flags=re.IGNORECASE)
    
    # Remove script content
    value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
    value = re.sub(r'on\w+\s*=', '', value, flags=re.IGNORECASE)
    
    # Limit length
    value = value[:max_length]
    
    return value.strip()

def validate_numeric_input(value: Any, param_name: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Validate numeric input with strict bounds checking"""
    
    # Convert to number
    try:
        if isinstance(value, str):
            # Check for malicious patterns in numeric strings
            if re.search(r'[^\d\.\-\+eE]', value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid numeric format for {param_name}"
                )
        
        numeric_value = float(value)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be a valid number"
        )
    
    # Check for reasonable bounds
    if abs(numeric_value) > MAX_NUMERIC_VALUE:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} value too large (max: {MAX_NUMERIC_VALUE})"
        )
    
    # Check custom bounds
    if min_val is not None and numeric_value < min_val:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be at least {min_val}"
        )
    
    if max_val is not None and numeric_value > max_val:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} cannot exceed {max_val}"
        )
    
    return numeric_value

@router.post("/generate", response_model=GenerationResponse)
async def generate_video(
    request: Union[GenerationRequest, DirectGenerationRequest, Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """
    Start video generation with Deforum Flux
    Supports both structured requests and flexible JSON payloads from frontend
    
    Args:
        request: Generation parameters and configuration (flexible format)
        
    Returns:
        Job ID and initial status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Handle different request formats
        if isinstance(request, dict):
            # Direct JSON payload from frontend
            parameters = request.get("parameters", {})
            config_data = request.get("config", {})
            motion_schedules = request.get("motion_schedules", {})
            
            # Merge motion schedules into parameters if provided
            if motion_schedules:
                parameters.update(motion_schedules)
                
        elif isinstance(request, DirectGenerationRequest):
            # DirectGenerationRequest format
            parameters = request.parameters
            config_data = request.config or {}
            motion_schedules = request.motion_schedules or {}
            
            # Merge motion schedules into parameters
            if motion_schedules:
                parameters.update(motion_schedules)
                
        elif isinstance(request, GenerationRequest):
            # Structured GenerationRequest format
            parameters = request.parameters.dict()
            config_data = request.config or {}
            
        else:
            # Try to extract from request object
            parameters = getattr(request, 'parameters', {})
            if hasattr(parameters, 'dict'):
                parameters = parameters.dict()
            config_data = getattr(request, 'config', {})
        
        # Extract prompt from parameters (handle both 'prompt' and 'prompts')
        prompt = parameters.get('prompts') or parameters.get('prompt', 'A beautiful landscape')
        
        logger.info(f"Starting generation job {job_id}", extra={
            "job_id": job_id,
            "prompt": prompt[:50] if prompt else "No prompt",
            "max_frames": parameters.get('max_frames', 30),
            "animation_mode": parameters.get('animation_mode', '2D')
        })
        
        # Create unified configuration
        unified_config = {
            **parameters,
            **config_data
        }
        
        # Validate and create job entry
        job_manager.create_job(job_id, {
            "parameters": parameters,
            "config": config_data,
            "unified_config": unified_config
        })
        
        # Start background generation
        background_tasks.add_task(process_generation, job_id, unified_config)
        
        return GenerationResponse(
            job_id=job_id,
            status="queued",
            message="Generation job started successfully"
        )
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail={
            "error": "Parameter validation failed",
            "details": e.validation_errors if hasattr(e, 'validation_errors') else str(e)
        })
    except Exception as e:
        logger.error(f"Generation request failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "Internal server error",
            "details": str(e)
        })

async def process_generation(job_id: str, config: Dict[str, Any]):
    """
    Process generation in background with comprehensive parameter support
    
    Args:
        job_id: Unique job identifier
        config: Unified configuration with all parameters
    """
    try:
        # Update job status
        job_manager.update_job(job_id, {
            "status": "processing",
            "message": "Initializing generation..."
        })
        
        # Parse and validate configuration
        parsed_config = parse_frontend_config(config)
        
        # Initialize bridge with parsed config
        bridge = FluxDeforumBridge(parsed_config)
        
        # Generate animation with progress updates
        frames = []
        total_frames = config.get('max_frames', 30)
        
        for frame_idx in range(total_frames):
            # Update progress
            progress = frame_idx / total_frames
            job_manager.update_job(job_id, {
                "status": "processing",
                "progress": progress,
                "current_frame": frame_idx,
                "total_frames": total_frames,
                "message": f"Generating frame {frame_idx + 1}/{total_frames}"
            })
            
            # Generate frame (placeholder - replace with actual generation)
            frame = await generate_single_frame(bridge, frame_idx, config)
            frames.append(frame)
            
            # Simulate processing time
            await asyncio.sleep(0.1)
        
        # Save results
        output_path = save_generation_result(job_id, frames, config)
        
        # Update job as completed
        job_manager.update_job(job_id, {
            "status": "completed",
            "progress": 1.0,
            "current_frame": total_frames,
            "message": "Generation completed successfully",
            "video_url": f"/api/v1/download/{job_id}",
            "output_path": output_path
        })
        
        logger.info(f"Generation job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Generation job {job_id} failed: {e}")
        job_manager.update_job(job_id, {
            "status": "error",
            "error": str(e),
            "message": f"Generation failed: {str(e)}"
        })

def parse_frontend_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse frontend configuration to bridge-compatible format
    
    Args:
        config: Raw configuration from frontend
        
    Returns:
        Parsed configuration compatible with FluxDeforumBridge
    """
    parsed = {}
    
    # Core generation parameters
    parsed['width'] = config.get('width', 1024)
    parsed['height'] = config.get('height', 1024)
    parsed['steps'] = config.get('num_inference_steps', config.get('steps', 20))
    parsed['guidance_scale'] = config.get('guidance_scale', 3.5)
    parsed['seed'] = config.get('seed', -1)
    parsed['max_frames'] = config.get('max_frames', 30)
    
    # Animation mode
    parsed['animation_mode'] = config.get('animation_mode', '2D')
    
    # Prompts (handle both formats)
    prompts = config.get('prompts') or config.get('prompt', 'A beautiful landscape')
    parsed['prompts'] = prompts
    parsed['prompt_2'] = config.get('prompt_2', '')
    
    # Motion schedules
    parsed['motion_schedule'] = parse_motion_schedules(config)
    
    # Performance settings
    # Note: quantization_type removed - using simple model loader without quantization
    parsed['enable_attention_slicing'] = config.get('enable_attention_slicing', False)
    parsed['enable_vae_tiling'] = config.get('enable_vae_tiling', False)
    parsed['offload'] = config.get('offload', True)
    
    # Output settings
    parsed['output_type'] = config.get('output_type', 'pil')
    parsed['batch_name'] = config.get('batch_name', 'deforum_flux')
    parsed['save_samples'] = config.get('save_samples', True)
    
    # Strength and noise schedules
    parsed['strength_schedule'] = config.get('strength_schedule', '0:(0.75)')
    parsed['noise_schedule'] = config.get('noise_schedule', '0:(0.02)')
    
    return parsed

def parse_motion_schedules(config: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """
    Parse motion schedules from keyframe strings to structured format
    
    Args:
        config: Configuration containing motion schedule strings
        
    Returns:
        Structured motion schedule
    """
    motion_schedule = {}
    
    # Motion parameters to parse
    motion_params = [
        'zoom', 'angle', 'translation_x', 'translation_y', 'translation_z',
        'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z'
    ]
    
    # Extract keyframes from all schedules
    all_frames = set([0])  # Always include frame 0
    
    for param in motion_params:
        schedule_str = config.get(param, '0:(0)')
        frames = extract_keyframes_from_schedule(schedule_str)
        all_frames.update(frames)
    
    # Build motion schedule for each frame
    for frame in sorted(all_frames):
        motion_schedule[frame] = {}
        for param in motion_params:
            schedule_str = config.get(param, '0:(0)' if param != 'zoom' else '0:(1.0)')
            value = extract_value_at_frame(schedule_str, frame)
            motion_schedule[frame][param] = value
    
    return motion_schedule

def extract_keyframes_from_schedule(schedule_str: str) -> List[int]:
    """Extract frame numbers from a keyframe schedule string"""
    frames = []
    if ',' in schedule_str:
        parts = schedule_str.split(',')
        for part in parts:
            if ':' in part:
                frame_part = part.split(':')[0].strip()
                try:
                    frames.append(int(frame_part))
                except ValueError:
                    continue
    else:
        if ':' in schedule_str:
            frame_part = schedule_str.split(':')[0].strip()
            try:
                frames.append(int(frame_part))
            except ValueError:
                pass
    
    return frames

def extract_value_at_frame(schedule_str: str, frame: int) -> float:
    """Extract value from keyframe schedule string at specific frame"""
    try:
        if ':' not in schedule_str:
            return 0.0
        
        # Handle multiple keyframes
        if ',' in schedule_str:
            parts = schedule_str.split(',')
            for part in parts:
                if ':' in part:
                    frame_part, value_part = part.split(':', 1)
                    if int(frame_part.strip()) == frame:
                        return float(value_part.strip().replace('(', '').replace(')', ''))
        else:
            # Single keyframe
            frame_part, value_part = schedule_str.split(':', 1)
            if int(frame_part.strip()) == frame:
                return float(value_part.strip().replace('(', '').replace(')', ''))
        
        # Default to first value if frame not found
        first_part = schedule_str.split(',')[0] if ',' in schedule_str else schedule_str
        if ':' in first_part:
            _, value_part = first_part.split(':', 1)
            return float(value_part.strip().replace('(', '').replace(')', ''))
            
    except (ValueError, IndexError):
        return 1.0 if 'zoom' in schedule_str else 0.0
    
    return 1.0 if 'zoom' in schedule_str else 0.0

async def generate_single_frame(bridge: FluxDeforumBridge, frame_idx: int, config: Dict[str, Any]):
    """Generate a single frame using the bridge"""
    # This is a placeholder - implement actual frame generation
    # In real implementation, this would call bridge.generate_frame()
    return f"frame_{frame_idx:04d}.png"

def save_generation_result(job_id: str, frames: List[str], config: Dict[str, Any]) -> str:
    """Save generation results to disk"""
    output_dir = Path(f"outputs/{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save frame list
    frames_file = output_dir / "frames.json"
    with open(frames_file, 'w') as f:
        import json
        json.dump({"frames": frames, "config": config}, f, indent=2)
    
    return str(output_dir)

@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of a generation job - Fixed error handling
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Current job status and progress
    """
    try:
        # Validate job_id format
        if not job_id or not re.match(r'^[a-f0-9-]{36}$', job_id):
            raise HTTPException(status_code=404, detail="Invalid job ID format")
        
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return StatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job.get("progress", 0.0),
            current_frame=job.get("current_frame", 0),
            total_frames=job.get("total_frames", 0),
            message=job.get("message", ""),
            error=job.get("error"),
            video_url=job.get("video_url"),
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at")
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404) as-is
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "Failed to retrieve job status",
            "details": str(e)
        })

@router.post("/validate")
async def validate_parameters(parameters: Dict[str, Any]):
    """
    Validate generation parameters with ENHANCED SECURITY input sanitization
    
    Args:
        parameters: Parameters to validate
        
    Returns:
        Validation results
    """
    try:
        errors = []
        warnings = []
        suggestions = []
        
        # Security validation - Check each parameter for malicious content
        for key, value in parameters.items():
            # Check parameter key for suspicious patterns
            if any(pattern in str(key).lower() for pattern in ['__', 'eval', 'exec', 'import', 'subprocess', 'os.system']):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Suspicious parameter name: {key}"
                )
            
            # Check string values for malicious content
            if isinstance(value, str) and is_malicious_input(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Malicious content detected in parameter: {key}"
                )
        
        # Sanitize and validate prompt
        if 'prompt' in parameters:
            try:
                sanitized_prompt = sanitize_string_input(parameters['prompt'], 2000)
                if len(parameters['prompt']) > 2000:
                    warnings.append("Prompt was truncated to 2000 characters")
                parameters['prompt'] = sanitized_prompt
            except HTTPException:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt contains malicious content"
                )
        
        # Validate dimensions with strict security bounds
        if 'width' in parameters:
            width = validate_numeric_input(parameters['width'], 'width', 64, 4096)
            if width % 64 != 0:
                raise HTTPException(status_code=400, detail="Width must be multiple of 64")
        
        if 'height' in parameters:
            height = validate_numeric_input(parameters['height'], 'height', 64, 4096)
            if height % 64 != 0:
                raise HTTPException(status_code=400, detail="Height must be multiple of 64")
        
        # Check total pixel count
        if 'width' in parameters and 'height' in parameters:
            total_pixels = parameters['width'] * parameters['height']
            if total_pixels > 4096 * 4096:
                raise HTTPException(
                    status_code=400,
                    detail="Image resolution too high - maximum 4096x4096"
                )
        
        # Validate frame count with strict bounds
        if 'max_frames' in parameters:
            max_frames = validate_numeric_input(parameters['max_frames'], 'max_frames', 1, 500)
            if max_frames > 100:
                warnings.append("High frame count will increase generation time")
        
        # Validate steps
        if 'steps' in parameters or 'num_inference_steps' in parameters:
            steps_value = parameters.get('steps', parameters.get('num_inference_steps', 20))
            steps = validate_numeric_input(steps_value, 'steps', 1, 100)
            if steps < 10:
                suggestions.append("Consider using at least 10 steps for better quality")
        
        # Validate guidance scale
        if 'guidance_scale' in parameters:
            validate_numeric_input(parameters['guidance_scale'], 'guidance_scale', 0.1, 20.0)
        
        # Validate animation mode
        if 'animation_mode' in parameters:
            mode = parameters['animation_mode']
            if mode not in ALLOWED_ANIMATION_MODES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid animation mode. Allowed: {ALLOWED_ANIMATION_MODES}"
                )
        
        return {
            "valid": True,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "parameter_count": len(parameters),
            "sanitized": True
        }
        
    except HTTPException:
        # Re-raise validation errors with proper 400 status
        raise
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        raise HTTPException(status_code=400, detail={
            "valid": False,
            "errors": ["Parameter validation failed - invalid input format"],
            "warnings": [],
            "suggestions": [],
            "parameter_count": 0
        })

@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """
    Download generation result
    
    Args:
        job_id: Job identifier
        
    Returns:
        File download response
    """
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        # Return download information
        return {
            "job_id": job_id,
            "status": "ready",
            "download_url": f"/api/v1/files/{job_id}/video.mp4",
            "message": "Generation ready for download"
        }
        
    except Exception as e:
        logger.error(f"Download failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "Download failed",
            "details": str(e)
        })
