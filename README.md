# Deforum Flux Backend

Flux backend for Deforum using Black Forest Labs Flux. This package includes code from [Black Forest Labs Flux](https://github.com/black-forest-labs/flux) to enable PyPI installation.

## Installation

```bash
pip install deforum-flux
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