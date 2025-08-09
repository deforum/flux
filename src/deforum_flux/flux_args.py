from pydantic import BaseModel, ConfigDict, model_validator
from PIL import Image
from typing import Optional, Union
import requests
from io import BytesIO
import torch

from .utils import center_crop_resize

class FluxArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prompt: str = "a beautiful sunset"
    width: int = 1024
    height: int = 1024
    num_steps: int = 50
    guidance: float = 3.5
    seed: Optional[int] = None
    img_cond: Optional[Union[str, Image.Image]] = None
    
    @model_validator(mode='after')
    def validate_and_process(self):
        # First, align dimensions to 16 pixel boundary for latent space
        self.height = 16 * (self.height // 16)
        self.width = 16 * (self.width // 16)
        
        # Process img_cond if provided (using aligned dimensions)
        if self.img_cond is not None:
            # Load image from various sources
            if isinstance(self.img_cond, Image.Image):
                img = self.img_cond
            elif isinstance(self.img_cond, str):
                if self.img_cond.startswith(('http://', 'https://')):
                    # It's a URL
                    response = requests.get(self.img_cond)
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    # It's a file path
                    img = Image.open(self.img_cond).convert("RGB")
            else:
                raise ValueError('img_cond must be a PIL Image, file path, or URL')
            
            # Apply center crop and resize using aligned dimensions
            self.img_cond = center_crop_resize(img, self.width, self.height)
        
        # Set default seed if not provided
        if self.seed is None:
            rng = torch.Generator(device="cpu")
            self.seed = rng.seed()
        
        return self