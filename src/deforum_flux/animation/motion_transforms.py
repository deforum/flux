"""
Motion Transforms for Classic Deforum 16-Channel Processing

This module provides geometric transformations and depth processing specifically
designed for 16-channel Flux latents in classic Deforum style.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from deforum.core.exceptions import MotionProcessingError, TensorProcessingError
from deforum.core.logging_config import get_logger
from deforum_flux.models.model_paths import get_model_path
from deforum.utils.device_utils import normalize_device, get_torch_device, ensure_tensor_device


class MotionTransforms:
    """Handles geometric transformations and depth processing for 16-channel latents."""
    
    def __init__(self, device: str = "cpu"):
        self.device = normalize_device(device)
        self.logger = get_logger(__name__)
        
        # Depth model will be initialized when needed
        self.depth_model = None
        self.depth_model_type = None
    
    def apply_geometric_transform(
        self, 
        latent: torch.Tensor, 
        motion_params: Dict[str, float]
    ) -> torch.Tensor:
        """
        Apply classic Deforum geometric transformations to 16-channel latent.
        
        Args:
            latent: 16-channel latent tensor (B, 16, H, W)
            motion_params: Motion parameters (zoom, angle, translation_x, translation_y, translation_z)
            
        Returns:
            Transformed latent tensor
            
        Raises:
            TensorProcessingError: If tensor shapes are invalid
            MotionProcessingError: If transformation fails
        """
        try:
            batch_size, channels, height, width = latent.shape
            
            if channels != 16:
                raise TensorProcessingError(
                    f"Expected 16-channel latent, got {channels} channels",
                    tensor_shape=latent.shape
                )
            
            # Extract motion parameters with defaults
            zoom = motion_params.get("zoom", 1.0)
            angle = motion_params.get("angle", 0.0)
            tx = motion_params.get("translation_x", 0.0)
            ty = motion_params.get("translation_y", 0.0)
            tz = motion_params.get("translation_z", 0.0)
            
            self.logger.debug(f"Applying geometric transform: zoom={zoom:.3f}, angle={angle:.1f}°, tx={tx:.1f}, ty={ty:.1f}, tz={tz:.1f}")
            
            # Apply 2D transformation first (existing functionality)
            transformed = self.apply_2d_transform(latent, zoom, angle, tx, ty)
            
            # Apply Z-axis transformation (depth morphing in latent space)
            if abs(tz) > 0.001:  # Only apply if significant Z movement
                transformed = self.apply_z_transform(transformed, tz)
            
            return transformed
            
        except Exception as e:
            raise MotionProcessingError(f"Geometric transformation failed: {e}")
    
    def apply_2d_transform(
        self,
        latent: torch.Tensor,
        zoom: float,
        angle: float, 
        tx: float,
        ty: float
    ) -> torch.Tensor:
        """
        Apply classic 2D affine transformation (zoom, rotate, translate).
        
        Args:
            latent: Input latent tensor
            zoom: Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
            angle: Rotation angle in degrees
            tx: Translation in X direction (pixels)
            ty: Translation in Y direction (pixels)
            
        Returns:
            Transformed latent tensor
        """
        batch_size, channels, height, width = latent.shape
        
        # Convert angle to radians
        angle_rad = torch.tensor(angle * np.pi / 180.0, device=latent.device)
        
        # Create transformation matrix
        cos_angle = torch.cos(angle_rad)
        sin_angle = torch.sin(angle_rad)
        
        # Affine transformation matrix for classic Deforum
        # [zoom*cos, -zoom*sin, tx]
        # [zoom*sin,  zoom*cos, ty]
        theta = torch.tensor([
            [zoom * cos_angle, -zoom * sin_angle, tx / width * 2],
            [zoom * sin_angle,  zoom * cos_angle, ty / height * 2]
        ], device=latent.device, dtype=latent.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Create sampling grid
        grid = F.affine_grid(theta, latent.size(), align_corners=False)
        
        # Apply transformation with reflection padding (classic Deforum behavior)
        transformed = F.grid_sample(
            latent, grid, 
            mode='bilinear',
            padding_mode='reflection', 
            align_corners=False
        )
        
        return transformed
    
    def apply_z_transform(
        self,
        latent: torch.Tensor,
        tz: float,
        use_real_depth: bool = False,
        decoded_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Z-axis transformation (depth morphing) in 16-channel latent space.
        
        This creates depth-like effects using either:
        1. Heuristic channel-wise transformations (fast, no external models needed)
        2. Real depth estimation with MiDaS/Depth Anything (accurate, requires models)
        
        Args:
            latent: 16-channel latent tensor (B, 16, H, W)
            tz: Z translation value (positive = closer, negative = further)
            use_real_depth: Whether to use real depth estimation models
            decoded_image: Decoded image for depth estimation (required if use_real_depth=True)
            
        Returns:
            Depth-transformed latent tensor
        """
        if use_real_depth and decoded_image is not None:
            return self._apply_real_depth_transform(latent, tz, decoded_image)
        else:
            return self._apply_heuristic_depth_transform(latent, tz)
    
    def _apply_heuristic_depth_transform(
        self,
        latent: torch.Tensor,
        tz: float
    ) -> torch.Tensor:
        """
        Apply heuristic depth transformation without external depth models.
        
        This creates realistic depth effects by:
        1. Scaling latent channels to simulate depth
        2. Applying channel-wise transformations for depth perception
        3. Creating smooth depth transitions with perspective effects
        """
        batch_size, channels, height, width = latent.shape
        
        # Normalize tz to reasonable range (-1.0 to 1.0)
        tz_normalized = torch.clamp(torch.tensor(tz / 100.0, device=latent.device), -1.0, 1.0)
        
        # Create depth scaling factor (closer = larger, further = smaller)
        depth_scale = 1.0 + (tz_normalized * 0.3)  # Max 30% scale change
        
        if abs(tz_normalized) < 0.001:
            return latent  # No significant movement
        
        # Method 1: Channel-wise depth transformation for realistic effect
        # Different channels respond differently to depth (simulate depth layers)
        channel_weights = torch.linspace(0.8, 1.2, channels, device=latent.device)
        channel_weights = channel_weights.view(1, channels, 1, 1)
        
        # Apply depth-aware channel scaling
        depth_effect = 1.0 + (tz_normalized * 0.2 * channel_weights)
        depth_transformed = latent * depth_effect
        
        # Method 2: Add perspective scaling for camera-like depth effect
        if abs(tz_normalized) > 0.1:
            # Create radial distance from center for perspective effect
            y_coords = torch.linspace(-1, 1, height, device=latent.device)
            x_coords = torch.linspace(-1, 1, width, device=latent.device)
            Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
            radial_dist = torch.sqrt(X**2 + Y**2)
            
            # Apply perspective scaling (stronger at edges, like real camera movement)
            perspective_factor = 1.0 + (tz_normalized * 0.15 * radial_dist)
            perspective_factor = perspective_factor.unsqueeze(0).unsqueeze(0)
            
            depth_transformed = depth_transformed * perspective_factor
        
        # Method 3: Add subtle depth-based blur simulation in latent space
        if abs(tz_normalized) > 0.2:
            # Apply slight blur to simulate depth of field
            kernel_size = 3
            sigma = abs(tz_normalized) * 0.5
            
            # Create Gaussian kernel
            kernel = self._create_gaussian_kernel(kernel_size, sigma, latent.device)
            kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
            
            # Apply grouped convolution (each channel processed separately)
            depth_transformed = F.conv2d(
                depth_transformed, kernel, 
                padding=kernel_size//2, groups=channels
            )
        
        self.logger.debug(f"Applied heuristic depth transform: tz={tz:.3f}, scale={depth_scale:.3f}")
        return depth_transformed
    
    def _apply_real_depth_transform(
        self,
        latent: torch.Tensor,
        tz: float,
        decoded_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply depth transformation using real depth estimation models.
        
        Args:
            latent: 16-channel latent tensor
            tz: Z translation value
            decoded_image: Decoded image for depth estimation
            
        Returns:
            Depth-transformed latent tensor
        """
        try:
            # Generate depth map
            depth_map = self._estimate_depth(decoded_image)
            
            # Convert single-channel depth to 16-channel latent space
            depth_latent = self._depth_to_16ch_latent(depth_map)
            
            # Apply depth-guided transformation
            return self._apply_depth_guided_transform(latent, depth_latent, tz)
            
        except Exception as e:
            self.logger.warning(f"Real depth transform failed, falling back to heuristic: {e}")
            return self._apply_heuristic_depth_transform(latent, tz)
    
    def _estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth using MiDaS or Depth Anything models.
        
        Args:
            image: RGB image tensor (B, 3, H, W)
            
        Returns:
            Depth map tensor (B, 1, H, W)
        """
        if self.depth_model is None:
            self._initialize_depth_model()
        
        if self.depth_model is None:
            raise MotionProcessingError("No depth model available")
        
        with torch.no_grad():
            # Ensure image is in correct format
            if image.dim() == 4 and image.shape[1] == 3:
                # Convert from [-1, 1] to [0, 1] if needed
                if image.min() < 0:
                    image = (image + 1) / 2
                
                # Apply depth model
                if self.depth_model_type == "midas":
                    depth = self._apply_midas_depth(image)
                elif self.depth_model_type == "depth_anything":
                    depth = self._apply_depth_anything(image)
                else:
                    raise MotionProcessingError(f"Unknown depth model type: {self.depth_model_type}")
                
                # Normalize to [0, 1]
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                
                return depth
            else:
                raise TensorProcessingError(f"Expected RGB image (B, 3, H, W), got {image.shape}")
    
    def _initialize_depth_model(self) -> None:
        """Initialize depth estimation model (MiDaS or Depth Anything)."""
        try:
            # Try MiDaS first (more stable)
            self._initialize_midas()
            if self.depth_model is not None:
                return
            
            # Fallback to Depth Anything
            self._initialize_depth_anything()
            if self.depth_model is not None:
                return
                
            self.logger.warning("No depth models could be initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize depth models: {e}")
    
    def _initialize_midas(self) -> None:
        """Initialize MiDaS depth model using centralized model paths."""
        try:
            import torch
            
            # Try to get centralized model path for MiDaS
            try:
                midas_path = get_model_path("midas")
                self.logger.info(f"Using centralized MiDaS path: {midas_path}")
                
                # Load from local path if available
                if midas_path and midas_path.exists():
                    self.depth_model = torch.jit.load(str(midas_path))
                    self.logger.info(f"Loaded MiDaS from centralized path: {midas_path}")
                else:
                    raise FileNotFoundError("MiDaS model not found in centralized path")
                    
            except Exception as path_error:
                self.logger.info(f"Centralized MiDaS path not available ({path_error}), using torch hub")
                
                # Fallback to torch hub
                model_type = "MiDaS_small"  # Faster, less memory
                self.depth_model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
                self.logger.info(f"MiDaS depth model loaded from torch hub: {model_type}")
            
            self.depth_model.to(get_torch_device(self.device))
            self.depth_model.eval()
            self.depth_model_type = "midas"
            
            self.logger.info("MiDaS depth model initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize MiDaS: {e}")
            self.depth_model = None
    
    def _initialize_depth_anything(self) -> None:
        """Initialize Depth Anything model using centralized model paths."""
        try:
            from transformers import pipeline
            
            # Try to get centralized model path for Depth Anything
            try:
                depth_anything_path = get_model_path("depth_anything")
                self.logger.info(f"Using centralized Depth Anything path: {depth_anything_path}")
                
                # Initialize pipeline with local path if available
                if depth_anything_path and depth_anything_path.exists():
                    self.depth_model = pipeline(
                        task="depth-estimation", 
                        model=str(depth_anything_path),
                        device=0 if normalize_device(self.device) == "cuda" else -1
                    )
                    self.logger.info(f"Loaded Depth Anything from centralized path: {depth_anything_path}")
                else:
                    raise FileNotFoundError("Depth Anything model not found in centralized path")
                    
            except Exception as path_error:
                self.logger.info(f"Centralized Depth Anything path not available ({path_error}), using HuggingFace hub")
                
                # Fallback to HuggingFace hub
                self.depth_model = pipeline(
                    task="depth-estimation", 
                    model="LiheYoung/depth-anything-small-hf",
                    device=0 if normalize_device(self.device) == "cuda" else -1
                )
                self.logger.info("Depth Anything model loaded from HuggingFace hub")
            
            self.depth_model_type = "depth_anything"
            self.logger.info("Depth Anything model initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Depth Anything: {e}")
            self.depth_model = None
    
    def _apply_midas_depth(self, image: torch.Tensor) -> torch.Tensor:
        """Apply MiDaS depth estimation."""
        batch_size = image.shape[0]
        depth_maps = []
        
        for i in range(batch_size):
            # MiDaS expects RGB image
            img = image[i]  # (3, H, W)
            
            # Apply MiDaS
            with torch.no_grad():
                depth = self.depth_model(img.unsqueeze(0))
                depth = depth.squeeze(0).unsqueeze(0)  # (1, H, W)
                depth_maps.append(depth)
        
        return torch.stack(depth_maps, dim=0)  # (B, 1, H, W)
    
    def _apply_depth_anything(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Depth Anything estimation."""
        # Convert tensor to PIL for transformers pipeline
        from torchvision.transforms import ToPILImage, ToTensor
        import numpy as np
        
        to_pil = ToPILImage()
        to_tensor = ToTensor()
        
        batch_size = image.shape[0]
        depth_maps = []
        
        for i in range(batch_size):
            # Convert to PIL Image
            pil_img = to_pil(image[i])
            
            # Apply Depth Anything
            result = self.depth_model(pil_img)
            depth_array = np.array(result["depth"])
            
            # Convert back to tensor
            depth_tensor = torch.from_numpy(depth_array).float().unsqueeze(0)  # (1, H, W)
            depth_maps.append(depth_tensor)
        
        return ensure_tensor_device(torch.stack(depth_maps, dim=0), self.device)  # (B, 1, H, W)
    
    def _depth_to_16ch_latent(self, depth_map: torch.Tensor, target_channels: int = 16) -> torch.Tensor:
        """
        Convert single-channel depth map to 16-channel latent representation.
        
        Args:
            depth_map: Single-channel depth map (B, 1, H, W)
            target_channels: Target number of channels (16 for Flux)
            
        Returns:
            16-channel depth latent (B, 16, H, W)
        """
        batch_size, _, height, width = depth_map.shape
        
        # Method 1: Channel-wise scaling with depth interpretation
        # Different channels represent different depth layers
        channel_scales = torch.linspace(0.8, 1.2, target_channels, device=depth_map.device)
        channel_scales = channel_scales.view(1, target_channels, 1, 1)
        
        # Expand depth map and apply channel-specific scaling
        depth_latent = depth_map.repeat(1, target_channels, 1, 1) * channel_scales
        
        # Method 2: Add depth-based channel variations
        # Create complementary depth representations
        inverted_depth = 1.0 - depth_map
        
        # Assign different depth interpretations to different channel groups
        depth_latent[:, :4] *= depth_map.repeat(1, 4, 1, 1)      # Close objects
        depth_latent[:, 4:8] *= inverted_depth.repeat(1, 4, 1, 1)  # Far objects
        depth_latent[:, 8:12] *= (depth_map * 0.5 + 0.5).repeat(1, 4, 1, 1)  # Mid-range
        depth_latent[:, 12:16] *= torch.sigmoid(depth_map * 2 - 1).repeat(1, 4, 1, 1)  # Smooth transitions
        
        return depth_latent
    
    def _apply_depth_guided_transform(
        self, 
        latent: torch.Tensor, 
        depth_latent: torch.Tensor, 
        tz: float
    ) -> torch.Tensor:
        """
        Apply depth-guided transformation using real depth information.
        
        Args:
            latent: Original 16-channel latent
            depth_latent: 16-channel depth representation
            tz: Z translation value
            
        Returns:
            Depth-guided transformed latent
        """
        # Normalize tz
        tz_normalized = tz / 100.0
        
        # Apply depth-aware scaling
        # Closer objects (higher depth values) move more with Z translation
        depth_factor = 1.0 + (tz_normalized * depth_latent * 0.5)
        transformed = latent * depth_factor
        
        # Apply depth-based perspective warping
        if abs(tz_normalized) > 0.1:
            # Create perspective grid based on depth
            batch_size, channels, height, width = latent.shape
            
            # Use depth information to create non-uniform scaling
            scale_factor = 1.0 + (tz_normalized * depth_latent.mean(dim=1, keepdim=True) * 0.3)
            transformed = transformed * scale_factor
        
        return transformed
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create a Gaussian kernel for blur effects."""
        coords = torch.arange(kernel_size, device=device, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        return g.unsqueeze(0) * g.unsqueeze(1)
    
    def get_available_depth_models(self) -> Dict[str, bool]:
        """
        Check which depth models are available.
        
        Returns:
            Dictionary indicating model availability
        """
        available = {"midas": False, "depth_anything": False}
        
        # Check MiDaS
        try:
            import torch
            torch.hub.list("intel-isl/MiDaS")
            available["midas"] = True
        except:
            pass
        
        # Check Depth Anything
        try:
            from transformers import pipeline
            available["depth_anything"] = True
        except:
            pass
        
        return available