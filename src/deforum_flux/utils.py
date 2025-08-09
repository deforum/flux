import torch
from einops import rearrange
from PIL import Image

def save_image(filename: str, x: torch.Tensor):
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(filename, quality=95, subsampling=0)

def center_crop_resize(image, target_width, target_height):
    """Center crop to target aspect ratio, then resize to target dimensions."""
    w, h = image.size
    target_aspect = target_width / target_height
    
    if w / h > target_aspect:
        # Crop width
        new_w = int(h * target_aspect)
        image = image.crop(((w - new_w) // 2, 0, (w + new_w) // 2, h))
    else:
        # Crop height
        new_h = int(w / target_aspect)
        image = image.crop((0, (h - new_h) // 2, w, (h + new_h) // 2))
    
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)