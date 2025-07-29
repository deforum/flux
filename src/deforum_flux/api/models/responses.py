"""
Response models for API endpoints
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class GenerationResponse(BaseModel):
    """Response for generation request"""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Response message")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    queue_position: Optional[int] = Field(default=None, description="Position in queue")

class StatusResponse(BaseModel):
    """Response for job status request"""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    progress: float = Field(..., description="Progress percentage (0.0-1.0)", ge=0.0, le=1.0)
    current_frame: int = Field(default=0, description="Current frame being processed", ge=0)
    total_frames: int = Field(default=0, description="Total frames to generate", ge=0)
    message: str = Field(default="", description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Results
    video_url: Optional[str] = Field(default=None, description="URL to download completed video")
    preview_frames: List[str] = Field(default=[], description="URLs to preview frames")
    
    # Timing
    started_at: Optional[datetime] = Field(default=None, description="Job start time")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion time")
    estimated_remaining: Optional[float] = Field(default=None, description="Estimated seconds remaining")
    
    # Performance metrics
    frames_per_second: Optional[float] = Field(default=None, description="Generation rate")
    memory_usage: Optional[Dict[str, float]] = Field(default=None, description="Memory usage stats")
    gpu_utilization: Optional[float] = Field(default=None, description="GPU utilization percentage")

class ValidationResponse(BaseModel):
    """Response for parameter validation"""
    
    valid: bool = Field(..., description="Whether parameters are valid")
    errors: List[str] = Field(default=[], description="Validation error messages")
    warnings: List[str] = Field(default=[], description="Validation warnings")
    suggestions: List[str] = Field(default=[], description="Parameter suggestions")
    
    # Validation details
    parameter_count: int = Field(default=0, description="Number of parameters validated")
    estimated_memory: Optional[float] = Field(default=None, description="Estimated memory usage (MB)")
    estimated_time: Optional[float] = Field(default=None, description="Estimated generation time (seconds)")

class PresetApplicationResponse(BaseModel):
    """Response for preset application"""
    
    preset_applied: str = Field(..., description="Applied preset ID")
    preset_name: str = Field(..., description="Applied preset name")
    updated_parameters: Dict[str, Any] = Field(..., description="Updated parameters")
    changes_made: List[str] = Field(default=[], description="List of changes made")
    
class PresetCategoryResponse(BaseModel):
    """Response for preset category"""
    
    name: str = Field(..., description="Category name")
    count: int = Field(..., description="Number of presets in category")
    description: str = Field(..., description="Category description")

class PresetTagResponse(BaseModel):
    """Response for preset tag"""
    
    name: str = Field(..., description="Tag name")
    count: int = Field(..., description="Number of presets with this tag")

class SystemStatusResponse(BaseModel):
    """Response for system status"""
    
    status: str = Field(..., description="System status")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_count: int = Field(default=0, description="Number of available GPUs")
    
    # Memory information
    gpu_memory_used: Optional[int] = Field(default=None, description="GPU memory used (bytes)")
    gpu_memory_total: Optional[int] = Field(default=None, description="Total GPU memory (bytes)")
    system_memory_used: Optional[int] = Field(default=None, description="System memory used (bytes)")
    system_memory_total: Optional[int] = Field(default=None, description="Total system memory (bytes)")
    
    # Performance metrics
    active_jobs: int = Field(default=0, description="Number of active generation jobs")
    completed_jobs: int = Field(default=0, description="Number of completed jobs")
    failed_jobs: int = Field(default=0, description="Number of failed jobs")
    average_generation_time: Optional[float] = Field(default=None, description="Average generation time per frame")
    
    # Server information
    server_time: datetime = Field(..., description="Current server time")
    uptime: Optional[float] = Field(default=None, description="Server uptime in seconds")
    version: str = Field(default="1.0.0", description="API version")

class JobListResponse(BaseModel):
    """Response for job listing"""
    
    jobs: List[StatusResponse] = Field(..., description="List of jobs")
    total_count: int = Field(..., description="Total number of jobs")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Number of jobs per page")
    has_next: bool = Field(default=False, description="Whether there are more pages")

class ErrorResponse(BaseModel):
    """Standard error response"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier for debugging")

class ProgressUpdate(BaseModel):
    """WebSocket progress update message"""
    
    job_id: str = Field(..., description="Job identifier")
    type: str = Field(..., description="Update type")
    progress: float = Field(..., description="Progress percentage", ge=0.0, le=1.0)
    current_frame: int = Field(..., description="Current frame number")
    total_frames: int = Field(..., description="Total frames")
    message: str = Field(default="", description="Progress message")
    
    # Frame data
    preview_frame: Optional[str] = Field(default=None, description="Base64 encoded preview frame")
    frame_time: Optional[float] = Field(default=None, description="Time taken for current frame")
    
    # Performance data
    frames_per_second: Optional[float] = Field(default=None, description="Current generation rate")
    memory_usage: Optional[float] = Field(default=None, description="Current memory usage")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Update timestamp")

class WebSocketMessage(BaseModel):
    """Generic WebSocket message"""
    
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default={}, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    
class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Health status")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_count: int = Field(default=0, description="Number of GPUs")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")

# Model configurations for better OpenAPI documentation
class Config:
    schema_extra = {
        "example": {
            "job_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing",
            "progress": 0.45,
            "current_frame": 14,
            "total_frames": 30,
            "message": "Generating frame 14/30"
        }
    }

# Apply example configurations
StatusResponse.Config = Config
GenerationResponse.Config = type('Config', (), {
    'schema_extra': {
        "example": {
            "job_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "queued",
            "message": "Generation job started successfully",
            "queue_position": 2
        }
    }
})()

ValidationResponse.Config = type('Config', (), {
    'schema_extra': {
        "example": {
            "valid": True,
            "errors": [],
            "warnings": ["High step count may increase generation time"],
            "suggestions": ["Consider reducing steps to 20 for faster generation"],
            "parameter_count": 15,
            "estimated_memory": 8192.5,
            "estimated_time": 45.2
        }
    }
})()