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

## Publish
```bash
python -m build
python -m twine upload dist/*
```

## License

- **Deforum Flux Backend wrapper**: MIT License
- **Flux code**: Apache 2.0 License (see `src/flux/LICENSE`)
- **FLUX.1-schnell model**: See `src/flux/LICENSE-FLUX1-schnell.md`
- **FLUX.1-dev model**: See `src/flux/LICENSE-FLUX1-dev.md`

**Important:** The FLUX.1-dev model has a non-commercial license. Check the license files before using in commercial applications.