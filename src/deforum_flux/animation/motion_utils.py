"""
Motion Utilities for Classic Deforum 16-Channel Processing

This module provides utility functions for motion processing, validation,
and analysis of 16-channel Flux latents.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from deforum.core.exceptions import TensorProcessingError, MotionProcessingError
from deforum.core.logging_config import get_logger
from deforum.utils.device_utils import normalize_device, get_torch_device, ensure_tensor_device


class MotionUtils:
    """Utility functions for motion processing and latent analysis."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_latent(self, latent: torch.Tensor, device: str) -> None:
        """
        Validate that a latent tensor is suitable for 16-channel processing.
        
        Args:
            latent: Latent tensor to validate
            device: Expected device
            
        Raises:
            TensorProcessingError: If validation fails
        """
        if not isinstance(latent, torch.Tensor):
            raise TensorProcessingError("Input must be a torch.Tensor")
        
        if latent.ndim != 4:
            raise TensorProcessingError(
                f"Expected 4D tensor (B, C, H, W), got {latent.ndim}D",
                tensor_shape=latent.shape
            )
        
        if latent.shape[1] != 16:
            raise TensorProcessingError(
                f"Expected 16 channels, got {latent.shape[1]}",
                tensor_shape=latent.shape,
                expected_shape=(latent.shape[0], 16, latent.shape[2], latent.shape[3])
            )
        
        # Normalize both devices for comparison
        tensor_device = normalize_device(str(latent.device))
        expected_device = normalize_device(device)
        
        if tensor_device != expected_device and expected_device != "cpu":
            raise TensorProcessingError(
                f"Tensor device {latent.device} doesn't match expected device {device}"
            )
        
        # Check for NaN or infinite values
        if torch.isnan(latent).any():
            raise TensorProcessingError("Latent contains NaN values")
        
        if torch.isinf(latent).any():
            raise TensorProcessingError("Latent contains infinite values")
        
        # Check reasonable value ranges
        if latent.abs().max() > 100.0:
            self.logger.warning(f"Latent contains very large values (max: {latent.abs().max():.2f})")
        
        self.logger.debug(f"Latent validation passed: {latent.shape}")
    
    def get_motion_statistics(self, latent: torch.Tensor) -> Dict[str, Any]:
        """
        Get comprehensive statistical information about a 16-channel latent tensor.
        
        Args:
            latent: 16-channel latent tensor (B, 16, H, W)
            
        Returns:
            Dictionary with detailed statistical information
        """
        with torch.no_grad():
            stats = {}
            
            # Basic tensor info
            stats["shape"] = list(latent.shape)
            stats["dtype"] = str(latent.dtype)
            stats["device"] = str(latent.device)
            
            # Overall statistics
            stats["overall"] = {
                "mean": latent.mean().item(),
                "std": latent.std().item(),
                "min": latent.min().item(),
                "max": latent.max().item(),
                "median": latent.median().item(),
                "abs_mean": latent.abs().mean().item()
            }
            
            # Per-channel statistics
            channel_means = latent.mean(dim=(0, 2, 3))
            channel_stds = latent.std(dim=(0, 2, 3))
            channel_mins = latent.min(dim=2)[0].min(dim=2)[0].mean(dim=0)  # Average across batch
            channel_maxs = latent.max(dim=2)[0].max(dim=2)[0].mean(dim=0)  # Average across batch
            
            stats["per_channel"] = {
                "means": channel_means.tolist(),
                "stds": channel_stds.tolist(),
                "mins": channel_mins.tolist(),
                "maxs": channel_maxs.tolist(),
                "mean_of_means": channel_means.mean().item(),
                "std_of_means": channel_means.std().item()
            }
            
            # Channel correlation analysis
            batch_size, channels, height, width = latent.shape
            flattened = latent.view(batch_size, channels, -1).mean(dim=0)  # Average across batch
            
            if channels > 1:
                try:
                    channel_corr = torch.corrcoef(flattened)
                    
                    # Remove diagonal (self-correlation)
                    mask = ~torch.eye(channels, dtype=bool)
                    off_diagonal_corr = channel_corr[mask]
                    
                    stats["channel_correlation"] = {
                        "matrix": channel_corr.tolist(),
                        "mean_correlation": off_diagonal_corr.mean().item(),
                        "max_correlation": off_diagonal_corr.max().item(),
                        "min_correlation": off_diagonal_corr.min().item(),
                        "std_correlation": off_diagonal_corr.std().item()
                    }
                except Exception as e:
                    stats["channel_correlation"] = {"error": str(e)}
            
            # Spatial statistics
            spatial_means = latent.mean(dim=1)  # Average across channels
            spatial_std = spatial_means.std(dim=(1, 2))  # Std across spatial dimensions
            
            stats["spatial"] = {
                "spatial_variance_mean": spatial_std.mean().item(),
                "spatial_variance_std": spatial_std.std().item(),
                "center_vs_edge_ratio": self._get_center_edge_ratio(latent)
            }
            
            # Motion analysis
            stats["motion_analysis"] = self._analyze_motion_potential(latent)
            
            return stats
    
    def compare_latents(
        self, 
        latent1: torch.Tensor, 
        latent2: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compare two latent tensors to analyze motion effects.
        
        Args:
            latent1: First latent tensor (e.g., original)
            latent2: Second latent tensor (e.g., after motion)
            
        Returns:
            Dictionary with comparison metrics
        """
        with torch.no_grad():
            comparison = {}
            
            # Basic difference metrics
            diff = latent2 - latent1
            comparison["difference"] = {
                "mean_absolute_difference": diff.abs().mean().item(),
                "root_mean_square_difference": (diff**2).mean().sqrt().item(),
                "max_absolute_difference": diff.abs().max().item(),
                "relative_change": (diff.abs().mean() / latent1.abs().mean()).item()
            }
            
            # Per-channel difference analysis
            channel_diff = diff.abs().mean(dim=(0, 2, 3))
            comparison["per_channel_difference"] = {
                "channel_differences": channel_diff.tolist(),
                "most_changed_channel": channel_diff.argmax().item(),
                "least_changed_channel": channel_diff.argmin().item(),
                "difference_variance": channel_diff.std().item()
            }
            
            # Spatial difference analysis
            spatial_diff = diff.abs().mean(dim=1)  # Average across channels
            comparison["spatial_difference"] = {
                "spatial_diff_mean": spatial_diff.mean().item(),
                "spatial_diff_std": spatial_diff.std().item(),
                "max_spatial_diff": spatial_diff.max().item()
            }
            
            # Motion direction analysis
            comparison["motion_direction"] = self._analyze_motion_direction(latent1, latent2)
            
            return comparison
    
    def _get_center_edge_ratio(self, latent: torch.Tensor) -> float:
        """Calculate ratio of center values to edge values."""
        batch_size, channels, height, width = latent.shape
        
        # Define center and edge regions
        center_h_start, center_h_end = height // 4, 3 * height // 4
        center_w_start, center_w_end = width // 4, 3 * width // 4
        
        # Get center and edge regions
        center = latent[:, :, center_h_start:center_h_end, center_w_start:center_w_end]
        
        # Edge regions (top, bottom, left, right)
        top = latent[:, :, :height//8, :]
        bottom = latent[:, :, -height//8:, :]
        left = latent[:, :, :, :width//8]
        right = latent[:, :, :, -width//8:]
        
        center_mean = center.abs().mean().item()
        edge_mean = torch.cat([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()]).abs().mean().item()
        
        return center_mean / (edge_mean + 1e-8)
    
    def _analyze_motion_potential(self, latent: torch.Tensor) -> Dict[str, Any]:
        """Analyze how suitable the latent is for motion processing."""
        analysis = {}
        
        # Gradient analysis (indicates detail level)
        grad_x = torch.abs(latent[:, :, :, 1:] - latent[:, :, :, :-1])
        grad_y = torch.abs(latent[:, :, 1:, :] - latent[:, :, :-1, :])
        
        analysis["gradient_strength"] = {
            "x_gradient_mean": grad_x.mean().item(),
            "y_gradient_mean": grad_y.mean().item(),
            "total_gradient_mean": (grad_x.mean() + grad_y.mean()).item() / 2
        }
        
        # Frequency analysis (indicates texture complexity)
        # Use 2D FFT to analyze frequency content
        batch_size, channels = latent.shape[:2]
        fft_analysis = []
        
        for b in range(min(batch_size, 2)):  # Limit to 2 samples for performance
            for c in range(min(channels, 4)):  # Limit to 4 channels for performance
                try:
                    fft = torch.fft.fft2(latent[b, c])
                    fft_magnitude = torch.abs(fft)
                    
                    # Low vs high frequency energy
                    h, w = fft_magnitude.shape
                    low_freq = fft_magnitude[:h//4, :w//4].mean().item()
                    high_freq = fft_magnitude[h//4:, w//4:].mean().item()
                    
                    fft_analysis.append({
                        "low_frequency_energy": low_freq,
                        "high_frequency_energy": high_freq,
                        "frequency_ratio": high_freq / (low_freq + 1e-8)
                    })
                except Exception:
                    continue
        
        if fft_analysis:
            analysis["frequency_analysis"] = {
                "mean_low_freq": np.mean([f["low_frequency_energy"] for f in fft_analysis]),
                "mean_high_freq": np.mean([f["high_frequency_energy"] for f in fft_analysis]),
                "mean_freq_ratio": np.mean([f["frequency_ratio"] for f in fft_analysis])
            }
        
        # Motion suitability score (0-1, higher is better for motion)
        gradient_score = min(analysis["gradient_strength"]["total_gradient_mean"] / 0.1, 1.0)
        
        frequency_score = 0.5
        if "frequency_analysis" in analysis:
            # Balance between low and high frequency content is good for motion
            freq_ratio = analysis["frequency_analysis"]["mean_freq_ratio"]
            frequency_score = 1.0 - abs(freq_ratio - 0.5) / 0.5  # Optimal around 0.5
        
        analysis["motion_suitability_score"] = (gradient_score + frequency_score) / 2
        
        return analysis
    
    def _analyze_motion_direction(
        self, 
        latent1: torch.Tensor, 
        latent2: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze the direction and type of motion between two latents."""
        diff = latent2 - latent1
        
        # Analyze spatial shifts
        batch_size, channels, height, width = diff.shape
        
        # Calculate center of mass shift
        y_coords = torch.arange(height, device=diff.device, dtype=torch.float32).view(-1, 1)
        x_coords = torch.arange(width, device=diff.device, dtype=torch.float32).view(1, -1)
        
        # Weight by absolute difference
        weights = diff.abs().mean(dim=1)  # Average across channels
        
        total_weight = weights.sum(dim=(1, 2), keepdim=True) + 1e-8
        
        # Calculate weighted center of mass
        y_center = (weights * y_coords).sum(dim=(1, 2)) / total_weight.squeeze()
        x_center = (weights * x_coords).sum(dim=(1, 2)) / total_weight.squeeze()
        
        # Reference center
        ref_y, ref_x = height / 2, width / 2
        
        direction = {
            "center_shift_y": (y_center - ref_y).mean().item(),
            "center_shift_x": (x_center - ref_x).mean().item(),
            "shift_magnitude": torch.sqrt((y_center - ref_y)**2 + (x_center - ref_x)**2).mean().item()
        }
        
        # Analyze scaling (zoom) effects
        # Compare variance before and after
        var1 = latent1.var(dim=(2, 3)).mean()
        var2 = latent2.var(dim=(2, 3)).mean()
        direction["scale_change"] = (var2 / (var1 + 1e-8)).item()
        
        # Analyze rotation effects (simplified)
        # Look for asymmetric changes
        left_half = diff[:, :, :, :width//2].abs().mean()
        right_half = diff[:, :, :, width//2:].abs().mean()
        top_half = diff[:, :, :height//2, :].abs().mean()
        bottom_half = diff[:, :, height//2:, :].abs().mean()
        
        direction["asymmetry"] = {
            "horizontal_asymmetry": abs(left_half - right_half).item(),
            "vertical_asymmetry": abs(top_half - bottom_half).item()
        }
        
        return direction
    
    def create_motion_mask(
        self, 
        latent: torch.Tensor, 
        motion_type: str = "uniform"
    ) -> torch.Tensor:
        """
        Create a motion mask for selective motion application.
        
        Args:
            latent: Input latent tensor
            motion_type: Type of motion mask ("uniform", "center", "edges", "gradient")
            
        Returns:
            Motion mask tensor (same shape as latent)
        """
        batch_size, channels, height, width = latent.shape
        
        if motion_type == "uniform":
            return torch.ones_like(latent)
        
        elif motion_type == "center":
            # Stronger motion in center, weaker at edges
            y_coords = torch.linspace(-1, 1, height, device=latent.device)
            x_coords = torch.linspace(-1, 1, width, device=latent.device)
            Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Gaussian-like falloff from center
            dist_from_center = torch.sqrt(X**2 + Y**2)
            mask = torch.exp(-dist_from_center**2 / 0.5)
            
            return mask.unsqueeze(0).unsqueeze(0).expand_as(latent)
        
        elif motion_type == "edges":
            # Stronger motion at edges, weaker in center
            y_coords = torch.linspace(-1, 1, height, device=latent.device)
            x_coords = torch.linspace(-1, 1, width, device=latent.device)
            Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            dist_from_center = torch.sqrt(X**2 + Y**2)
            mask = torch.clamp(dist_from_center, 0, 1)
            
            return mask.unsqueeze(0).unsqueeze(0).expand_as(latent)
        
        elif motion_type == "gradient":
            # Motion strength based on gradient (more motion where there's more detail)
            grad_x = torch.abs(latent[:, :, :, 1:] - latent[:, :, :, :-1])
            grad_y = torch.abs(latent[:, :, 1:, :] - latent[:, :, :-1, :])
            
            # Pad gradients to match original size
            grad_x = torch.cat([grad_x, grad_x[:, :, :, -1:]], dim=3)
            grad_y = torch.cat([grad_y, grad_y[:, :, -1:, :]], dim=2)
            
            # Combined gradient magnitude
            gradient_magnitude = grad_x + grad_y
            
            # Normalize to [0, 1]
            mask = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
            
            return mask
        
        else:
            raise ValueError(f"Unknown motion_type: {motion_type}")
    
    def interpolate_latents(
        self, 
        latent1: torch.Tensor, 
        latent2: torch.Tensor, 
        num_steps: int,
        interpolation_mode: str = "linear"
    ) -> List[torch.Tensor]:
        """
        Interpolate between two latents for smooth transitions.
        
        Args:
            latent1: Starting latent
            latent2: Ending latent
            num_steps: Number of interpolation steps
            interpolation_mode: "linear", "cubic", or "slerp"
            
        Returns:
            List of interpolated latents
        """
        if interpolation_mode == "linear":
            alphas = torch.linspace(0, 1, num_steps, device=latent1.device)
            return [
                latent1 * (1 - alpha) + latent2 * alpha 
                for alpha in alphas
            ]
        
        elif interpolation_mode == "cubic":
            # Smooth cubic interpolation
            t = torch.linspace(0, 1, num_steps, device=latent1.device)
            alphas = 3 * t**2 - 2 * t**3  # Cubic smoothstep
            return [
                latent1 * (1 - alpha) + latent2 * alpha 
                for alpha in alphas
            ]
        
        elif interpolation_mode == "slerp":
            # Spherical linear interpolation
            # Normalize latents first
            latent1_norm = latent1 / (latent1.norm(dim=(1, 2, 3), keepdim=True) + 1e-8)
            latent2_norm = latent2 / (latent2.norm(dim=(1, 2, 3), keepdim=True) + 1e-8)
            
            # Calculate angle between vectors
            dot_product = (latent1_norm * latent2_norm).sum(dim=(1, 2, 3), keepdim=True)
            dot_product = torch.clamp(dot_product, -1, 1)
            omega = torch.acos(dot_product)
            
            interpolated = []
            for i in range(num_steps):
                t = i / (num_steps - 1) if num_steps > 1 else 0
                
                # SLERP formula
                sin_omega = torch.sin(omega)
                if sin_omega.abs().min() > 1e-6:
                    a = torch.sin((1 - t) * omega) / sin_omega
                    b = torch.sin(t * omega) / sin_omega
                    result = a * latent1_norm + b * latent2_norm
                else:
                    # Fallback to linear interpolation if vectors are nearly parallel
                    result = (1 - t) * latent1_norm + t * latent2_norm
                
                # Restore original magnitude
                original_mag1 = latent1.norm(dim=(1, 2, 3), keepdim=True)
                original_mag2 = latent2.norm(dim=(1, 2, 3), keepdim=True)
                target_mag = (1 - t) * original_mag1 + t * original_mag2
                
                result = result * target_mag
                interpolated.append(result)
            
            return interpolated
        
        else:
            raise ValueError(f"Unknown interpolation_mode: {interpolation_mode}")
    
    def optimize_motion_parameters(
        self,
        latent: torch.Tensor,
        target_motion: str = "smooth"
    ) -> Dict[str, float]:
        """
        Suggest optimal motion parameters based on latent characteristics.
        
        Args:
            latent: Input latent tensor
            target_motion: Type of desired motion ("smooth", "dynamic", "subtle")
            
        Returns:
            Suggested motion parameters
        """
        stats = self.get_motion_statistics(latent)
        
        # Base parameters
        params = {
            "zoom": 1.0,
            "angle": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0
        }
        
        # Adjust based on motion suitability
        motion_score = stats["motion_analysis"]["motion_suitability_score"]
        
        if target_motion == "smooth":
            params.update({
                "zoom": 1.0 + motion_score * 0.02,  # 1-2% zoom
                "angle": motion_score * 1.0,        # Up to 1 degree rotation
                "translation_z": motion_score * 5.0  # Subtle depth movement
            })
        
        elif target_motion == "dynamic":
            params.update({
                "zoom": 1.0 + motion_score * 0.05,  # Up to 5% zoom
                "angle": motion_score * 3.0,        # Up to 3 degrees rotation
                "translation_x": motion_score * 10.0,  # Horizontal movement
                "translation_z": motion_score * 15.0   # More depth movement
            })
        
        elif target_motion == "subtle":
            params.update({
                "zoom": 1.0 + motion_score * 0.01,  # Very small zoom
                "angle": motion_score * 0.5,        # Very small rotation
                "translation_z": motion_score * 2.0  # Minimal depth
            })
        
        return params