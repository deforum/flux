import torch
from einops import rearrange
from PIL import Image

def save_image(filename: str, x: torch.Tensor):
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(filename, quality=95, subsampling=0)