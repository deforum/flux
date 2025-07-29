#!/usr/bin/env python3
"""
Model Path Configuration for Centralized Model Management
Provides configurable model paths that can reference workspace models or local copies.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union
import json


class ModelPathManager:
    """Manages model paths with support for workspace centralization and fallbacks."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self._paths = {}
        self._load_configuration()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        current_dir = Path(__file__).parent
        return str(current_dir / "model_paths.json")
    
    def _load_configuration(self):
        """Load model path configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self._paths = config.get('model_paths', {})
            except Exception as e:
                print(f"Warning: Failed to load model paths config: {e}")
                self._setup_default_paths()
        else:
            self._setup_default_paths()
            self._save_configuration()
    
    def _setup_default_paths(self):
        """Set up default model paths with workspace and fallback locations."""
        project_root = Path(__file__).parent.parent.parent.parent
        workspace_models = project_root.parent / "workspace" / "models"
        local_models = Path(__file__).parent.parent / "models"
        
        self._paths = {
            # Base model directories
            "workspace_models": str(workspace_models),
            "local_models": str(local_models),
            "use_workspace": True,
            
            # Specific model paths
            "clip": {
                "workspace": str(workspace_models / "clip"),
                "local": str(local_models / "clip"),
                "enabled": True
            },
            "t5": {
                "workspace": str(workspace_models / "t5"),
                "local": str(local_models / "t5"),
                "enabled": True
            },
            "unet": {
                "workspace": str(workspace_models / "unet"),
                "local": str(local_models / "unet"),
                "enabled": True
            },
            "vae": {
                "workspace": str(workspace_models / "vae"),
                "local": str(local_models / "vae"),
                "enabled": True
            },
            "flux": {
                "workspace": str(workspace_models / "flux"),
                "local": str(local_models / "flux"),
                "enabled": True
            },
            "motion": {
                "workspace": str(workspace_models / "motion"),
                "local": str(local_models / "motion"),
                "enabled": True
            }
        }
    
    def _save_configuration(self):
        """Save current configuration to file."""
        config = {
            "model_paths": self._paths,
            "description": "Centralized model path configuration",
            "last_updated": str(Path(__file__).stat().st_mtime)
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save model paths config: {e}")
    
    def get_model_path(self, model_type: str) -> str:
        """Get the active model path for a specific model type."""
        if model_type not in self._paths:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_config = self._paths[model_type]
        
        # If it's a string path (for base directories)
        if isinstance(model_config, str):
            return model_config
        
        # If it's a dict with workspace/local options
        if isinstance(model_config, dict):
            use_workspace = self._paths.get("use_workspace", True)
            
            if use_workspace and model_config.get("enabled", True):
                workspace_path = model_config.get("workspace")
                if workspace_path and os.path.exists(workspace_path):
                    return workspace_path
            
            # Fallback to local path
            local_path = model_config.get("local")
            if local_path:
                return local_path
            
            raise ValueError(f"No valid path found for model type: {model_type}")
        
        raise ValueError(f"Invalid model configuration for: {model_type}")
    
    def get_all_model_paths(self) -> Dict[str, str]:
        """Get all active model paths."""
        paths = {}
        for model_type in ["clip", "t5", "unet", "vae", "flux", "motion"]:
            try:
                paths[model_type] = self.get_model_path(model_type)
            except ValueError:
                pass  # Skip models without valid paths
        return paths
    
    def set_workspace_mode(self, use_workspace: bool):
        """Toggle between workspace and local model paths."""
        self._paths["use_workspace"] = use_workspace
        self._save_configuration()
    
    def add_model_path(self, model_type: str, workspace_path: str, local_path: str, enabled: bool = True):
        """Add or update a model path configuration."""
        self._paths[model_type] = {
            "workspace": str(workspace_path),
            "local": str(local_path),
            "enabled": enabled
        }
        self._save_configuration()
    
    def create_symbolic_links(self, force: bool = False) -> Dict[str, bool]:
        """Create symbolic links from local model directories to workspace."""
        results = {}
        
        for model_type in ["clip", "t5", "unet", "vae", "flux", "motion"]:
            try:
                model_config = self._paths.get(model_type, {})
                if not isinstance(model_config, dict):
                    continue
                
                workspace_path = Path(model_config.get("workspace", ""))
                local_path = Path(model_config.get("local", ""))
                
                if not workspace_path or not local_path:
                    continue
                
                # Create workspace directory if it doesn't exist
                workspace_path.mkdir(parents=True, exist_ok=True)
                
                # Remove existing local path if it exists and force is True
                if local_path.exists() and force:
                    if local_path.is_symlink():
                        local_path.unlink()
                    elif local_path.is_dir():
                        # Only remove if it's empty or force is requested
                        try:
                            local_path.rmdir()
                        except OSError:
                            if force:
                                import shutil
                                shutil.rmtree(local_path)
                            else:
                                results[model_type] = False
                                continue
                
                # Create symbolic link
                if not local_path.exists():
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.symlink_to(workspace_path, target_is_directory=True)
                    results[model_type] = True
                else:
                    results[model_type] = False  # Already exists
                    
            except Exception as e:
                print(f"Failed to create symlink for {model_type}: {e}")
                results[model_type] = False
        
        return results
    
    def validate_paths(self) -> Dict[str, Dict[str, bool]]:
        """Validate all configured model paths."""
        validation_results = {}
        
        for model_type in ["clip", "t5", "unet", "vae", "flux", "motion"]:
            model_config = self._paths.get(model_type, {})
            if not isinstance(model_config, dict):
                continue
            
            workspace_path = model_config.get("workspace")
            local_path = model_config.get("local")
            
            validation_results[model_type] = {
                "workspace_exists": bool(workspace_path and os.path.exists(workspace_path)),
                "local_exists": bool(local_path and os.path.exists(local_path)),
                "workspace_writable": bool(workspace_path and os.access(workspace_path, os.W_OK)) if workspace_path and os.path.exists(workspace_path) else False,
                "local_writable": bool(local_path and os.access(local_path, os.W_OK)) if local_path and os.path.exists(local_path) else False,
                "is_symlink": bool(local_path and Path(local_path).is_symlink()) if local_path else False
            }
        
        return validation_results
    
    def get_status_report(self) -> str:
        """Generate a status report of model path configuration."""
        report = ["Model Path Configuration Status", "=" * 40]
        
        report.append(f"Workspace Mode: {'Enabled' if self._paths.get('use_workspace', True) else 'Disabled'}")
        report.append(f"Workspace Root: {self._paths.get('workspace_models', 'Not set')}")
        report.append(f"Local Root: {self._paths.get('local_models', 'Not set')}")
        report.append("")
        
        validation = self.validate_paths()
        for model_type, status in validation.items():
            report.append(f"{model_type.upper()}:")
            report.append(f"  Workspace: {' ++[√]++' if status['workspace_exists'] else '==[X]=='} {self._paths.get(model_type, {}).get('workspace', 'Not configured')}")
            report.append(f"  Local: {' ++[√]++' if status['local_exists'] else '==[X]=='} {self._paths.get(model_type, {}).get('local', 'Not configured')}")
            report.append(f"  Symlink: {' ++[√]++' if status['is_symlink'] else '==[X]=='}")
            report.append("")
        
        return "\n".join(report)


# Global instance for easy access
model_paths = ModelPathManager()


def get_model_path(model_type: str) -> str:
    """Convenience function to get model path."""
    return model_paths.get_model_path(model_type)


def get_all_model_paths() -> Dict[str, str]:
    """Convenience function to get all model paths."""
    return model_paths.get_all_model_paths()


def setup_workspace_models(force: bool = False) -> Dict[str, bool]:
    """Convenience function to set up workspace model symlinks."""
    model_paths.set_workspace_mode(True)
    return model_paths.create_symbolic_links(force=force)


if __name__ == "__main__":
    # CLI interface for model path management
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            print(model_paths.get_status_report())
        elif command == "setup":
            force = "--force" in sys.argv
            results = setup_workspace_models(force=force)
            print("Symbolic link creation results:")
            for model_type, success in results.items():
                status = " ++[√]++ Created" if success else "==[X]== Failed/Exists"
                print(f"  {model_type}: {status}")
        elif command == "validate":
            validation = model_paths.validate_paths()
            print("Model path validation:")
            for model_type, status in validation.items():
                print(f"  {model_type}: {status}")
        else:
            print("Usage: python model_paths.py [status|setup|validate] [--force]")
    else:
        print(model_paths.get_status_report())