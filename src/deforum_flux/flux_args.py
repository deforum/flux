from pydantic import BaseModel, ConfigDict, model_validator
from PIL import Image
from typing import Optional, Union
import torch

from .utils import center_crop_resize, load_image_from_source

class FluxArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: str = "a beautiful sunset"
    width: int = 1024
    height: int = 1024
    num_steps: int = 50
    guidance: float = 3.5
    seed: Optional[int] = None

    # redux
    img_redux: Optional[Union[str, Image.Image]] = None
    redux_strength: float = 0.5
    redux_mask: float = 0.5

    # init
    init_image: Optional[Union[str, Image.Image]] = None
    gamma: float = 0.8
    eta: float = 0.2
    start_timestep: float = 1.0
    stop_timestep: float = 0.8

    # kontext
    img_kontext: Optional[Union[str, Image.Image]] = None
    
    @model_validator(mode='after')
    def validate_and_process(self):
        # First, align dimensions to 16 pixel boundary for latent space
        self.height = 16 * (self.height // 16)
        self.width = 16 * (self.width // 16)
        
        # Process img_redux if provided
        if self.img_redux is not None:
            img = load_image_from_source(self.img_redux, 'img_redux')
            self.img_redux = center_crop_resize(img, self.width, self.height)

        if self.img_kontext is not None:
            img = load_image_from_source(self.img_kontext, 'img_kontext')
            self.img_kontext = center_crop_resize(img, self.width, self.height)
        
        # Process init_image if provided
        if self.init_image is not None:
            img = load_image_from_source(self.init_image, 'init_image')
            self.init_image = center_crop_resize(img, self.width, self.height)
        
        # Set default seed if not provided
        if self.seed is None:
            rng = torch.Generator(device="cpu")
            self.seed = rng.seed()
        
        return self