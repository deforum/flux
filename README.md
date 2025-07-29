# Deforum Flux Backend

Flux backend for Deforum using Black Forest Labs Flux. This package includes code from [Black Forest Labs Flux](https://github.com/black-forest-labs/flux) to enable PyPI installation.

## Installation

Install PyTorch with CUDA 12.8 support first:

```bash
# Install PyTorch with CUDA 12.8 (required for RTX 50 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install deforum-flux
pip install deforum-flux

# Optional: For TensorRT acceleration
pip install deforum-flux[tensorrt]
```

**Note:** RTX 50 series cards require CUDA 12.8.

## Development Installation

```bash
# Clone the repository
git clone https://github.com/deforum/flux.git
cd flux

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install in development mode
pip install -e .

# Optional: Install with TensorRT support
pip install -e .[tensorrt]
```
## Structure

```
flux/src/deforum_flux/ (GENERATOR)
  ├── animation/                 
  │   ├── motion_engine.py       
  │   ├── motion_transforms.py   
  │   ├── motion_utils.py        
  │   └── parameter_engine.py
  ├── models/                 
  │   ├── model_paths.py
  │   ├── models.py
  │   └── model_manager.py
  ├── bridge/   
  │   ├── bridge_config.py
  │   ├── bridge_generation_utils.py
  │   ├── bridge_stats_and_cleanup.py
  │   └── dependency_config.py
  │   └── flux_deforum_bridge.py
  └── api/
  │   ├── routes/
  │   ├── models/

```


## Publish
```bash
python -m build
python -m twine upload dist/*
```

## License

This package includes code from multiple sources:

- **Deforum Flux Backend** (wrapper code): MIT License
- **Black Forest Labs Flux** (core implementation): Apache 2.0 License
- **FLUX.1-schnell model**: Apache 2.0 License (commercial use allowed)
- **FLUX.1-dev model**: Non-commercial license (no commercial use)

⚠️ **Important**: If you use the FLUX.1-dev model, you are bound by its non-commercial license terms. See `src/flux/LICENSE-FLUX1-dev.md` for details.

For commercial applications, use only the FLUX.1-schnell model.