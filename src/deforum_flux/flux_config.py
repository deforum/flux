from pydantic import BaseModel, field_validator
from typing import Optional
import torch

MODELS = [
    "flux-schnell",
    "flux-dev",
    "flux-dev-kontext",
    "flux-dev-redux"
]

class FluxConfig(BaseModel):
    name: str = "flux-schnell"
    num_steps: Optional[int] = None
    offload: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v):
        if v not in MODELS:
            raise ValueError(f"Model must be one of {MODELS}")
        return v