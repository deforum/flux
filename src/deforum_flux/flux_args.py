from pydantic import BaseModel, ConfigDict, model_validator
from PIL import Image
from typing import Optional, Union
import torch

from .utils import center_crop_resize, load_image_from_source

class FluxArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Existing parameters
    prompt: str = "a beautiful sunset"
    width: int = 1024
    height: int = 1024
    num_steps: int = 50
    guidance: float = 3.5
    seed: Optional[int] = None
    img_cond: Optional[Union[str, Image.Image]] = None
    
    # Init image parameters
    init_image: Optional[Union[str, Image.Image]] = None
    strength: float = 1.0  # 0.0 = pure init_image, 1.0 = pure noise
    
    # RF-Inversion parameters (for later)
    # do_inversion: bool = False
    # gamma: float = 0.5
    # eta: float = 1.0
    # start_step: int = 0
    # end_step: int = 5
    # inversion_steps: int = 28
    
    @model_validator(mode='after')
    def validate_and_process(self):
        # First, align dimensions to 16 pixel boundary for latent space
        self.height = 16 * (self.height // 16)
        self.width = 16 * (self.width // 16)
        
        # Process img_cond if provided
        if self.img_cond is not None:
            img = load_image_from_source(self.img_cond, 'img_cond')
            self.img_cond = center_crop_resize(img, self.width, self.height)
        
        # Process init_image if provided
        if self.init_image is not None:
            img = load_image_from_source(self.init_image, 'init_image')
            self.init_image = center_crop_resize(img, self.width, self.height)
        
        # Validation: RF-Inversion requires init_image
        if self.do_inversion and self.init_image is None:
            raise ValueError("init_image must be provided when do_inversion=True")
        
        # Set default seed if not provided
        if self.seed is None:
            rng = torch.Generator(device="cpu")
            self.seed = rng.seed()
        
        return self