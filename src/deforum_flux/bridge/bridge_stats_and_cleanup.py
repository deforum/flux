"""
Statistics and Resource Management for Flux-Deforum Bridge

This module handles performance statistics tracking and resource cleanup.
"""

import torch
import time
from typing import Dict, Any, Optional
from deforum.core.logging_config import get_logger


class BridgeStatsManager:
    """Manages performance statistics and resource cleanup for the bridge."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.stats = {
            "frames_generated": 0,
            "total_generation_time": 0.0,
            "average_frame_time": 0.0,
            "memory_peak": 0.0,
            "last_generation_time": 0.0,
            "animation_count": 0,
            "total_animation_time": 0.0,
            "average_animation_time": 0.0
        }
        self.start_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        current_stats = self.stats.copy()
        
        # Add runtime statistics
        current_stats["uptime_seconds"] = time.time() - self.start_time
        
        # Add memory information if available
        if torch.cuda.is_available():
            current_stats["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            current_stats["gpu_memory_reserved"] = torch.cuda.memory_reserved()
            current_stats["gpu_memory_cached"] = torch.cuda.memory_cached()
        
        return current_stats
    
    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        self.stats = {
            "frames_generated": 0,
            "total_generation_time": 0.0,
            "average_frame_time": 0.0,
            "memory_peak": 0.0,
            "last_generation_time": 0.0,
            "animation_count": 0,
            "total_animation_time": 0.0,
            "average_animation_time": 0.0
        }
        self.start_time = time.time()
        self.logger.info("Performance statistics reset")
    
    def update_frame_stats(self, generation_time: float) -> None:
        """
        Update statistics after frame generation.
        
        Args:
            generation_time: Time taken to generate the frame
        """
        self.stats["frames_generated"] += 1
        self.stats["total_generation_time"] += generation_time
        self.stats["last_generation_time"] = generation_time
        
        # Update average
        if self.stats["frames_generated"] > 0:
            self.stats["average_frame_time"] = (
                self.stats["total_generation_time"] / self.stats["frames_generated"]
            )
        
        # Update memory peak if available
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            if current_memory > self.stats["memory_peak"]:
                self.stats["memory_peak"] = current_memory
    
    def update_animation_stats(self, animation_time: float, frame_count: int) -> None:
        """
        Update statistics after animation generation.
        
        Args:
            animation_time: Total time for animation generation
            frame_count: Number of frames in the animation
        """
        self.stats["animation_count"] += 1
        self.stats["total_animation_time"] += animation_time
        
        # Update average
        if self.stats["animation_count"] > 0:
            self.stats["average_animation_time"] = (
                self.stats["total_animation_time"] / self.stats["animation_count"]
            )
        
        self.logger.info(f"Animation completed: {frame_count} frames in {animation_time:.2f}s")
    
    def log_performance_summary(self) -> None:
        """Log a summary of current performance statistics."""
        stats = self.get_stats()
        
        self.logger.info("Performance Summary:", extra={
            "frames_generated": stats["frames_generated"],
            "average_frame_time": f"{stats['average_frame_time']:.3f}s",
            "animations_generated": stats["animation_count"],
            "uptime": f"{stats['uptime_seconds']:.1f}s"
        })
        
        if torch.cuda.is_available():
            gpu_memory_gb = stats.get("gpu_memory_allocated", 0) / (1024**3)
            self.logger.info(f"GPU Memory Usage: {gpu_memory_gb:.2f} GB")
    
    def cleanup_resources(self, memory_efficient: bool = True) -> None:
        """
        Clean up resources and free memory.
        
        Args:
            memory_efficient: Whether to perform aggressive cleanup
        """
        cleaned_items = []
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            
            if memory_efficient:
                # More aggressive cleanup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            memory_freed = initial_memory - final_memory
            
            if memory_freed > 0:
                cleaned_items.append(f"GPU memory: {memory_freed / (1024**2):.1f} MB freed")
        
        # Log cleanup results
        if cleaned_items:
            self.logger.info("Resource cleanup completed: " + ", ".join(cleaned_items))
        else:
            self.logger.debug("Resource cleanup completed (no significant cleanup needed)")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get detailed memory information.
        
        Returns:
            Dictionary with memory usage details
        """
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_available": True,
                "gpu_device_count": torch.cuda.device_count(),
                "gpu_current_device": torch.cuda.current_device(),
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
                "gpu_memory_cached": torch.cuda.memory_cached(),
                "gpu_max_memory_allocated": torch.cuda.max_memory_allocated(),
                "gpu_max_memory_reserved": torch.cuda.max_memory_reserved()
            })
            
            # Add device properties
            device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
            memory_info.update({
                "gpu_name": device_props.name,
                "gpu_total_memory": device_props.total_memory,
                "gpu_major": device_props.major,
                "gpu_minor": device_props.minor
            })
        else:
            memory_info["gpu_available"] = False
        
        return memory_info
    
    def monitor_memory_usage(self, operation_name: str) -> "MemoryMonitor":
        """
        Create a context manager for monitoring memory usage during an operation.
        
        Args:
            operation_name: Name of the operation being monitored
            
        Returns:
            MemoryMonitor context manager
        """
        return MemoryMonitor(operation_name, self.logger)


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""
    
    def __init__(self, operation_name: str, logger):
        self.operation_name = operation_name
        self.logger = logger
        self.start_memory = 0
        self.start_time = 0
    
    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_delta = end_memory - self.start_memory
            
            self.logger.debug(f"{self.operation_name} completed", extra={
                "duration": f"{duration:.3f}s",
                "memory_delta": f"{memory_delta / (1024**2):.1f} MB",
                "final_memory": f"{end_memory / (1024**2):.1f} MB"
            })
        else:
            self.logger.debug(f"{self.operation_name} completed in {duration:.3f}s")


class ResourceManager:
    """Manages system resources and provides cleanup utilities."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def cleanup_all(self) -> None:
        """Perform comprehensive resource cleanup."""
        self.logger.info("Starting comprehensive resource cleanup")
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Second pass for thorough cleanup
        
        # Python garbage collection
        import gc
        gc.collect()
        
        self.logger.info("Comprehensive resource cleanup completed")
    
    def check_system_resources(self) -> Dict[str, Any]:
        """
        Check available system resources.
        
        Returns:
            Dictionary with system resource information
        """
        resources = {}
        
        # GPU resources
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            resources["gpu"] = {
                "available": True,
                "device_count": device_count,
                "devices": []
            }
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                resources["gpu"]["devices"].append({
                    "index": i,
                    "name": props.name,
                    "memory_total": props.total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i)
                })
        else:
            resources["gpu"] = {"available": False}
        
        # CPU and system memory
        try:
            import psutil
            resources["system"] = {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "memory_percent": psutil.virtual_memory().percent
            }
        except ImportError:
            resources["system"] = {"available": False, "reason": "psutil not installed"}
        
        return resources