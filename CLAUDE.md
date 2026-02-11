# CLAUDE.md - AI Assistant Guide for SHINE

## Project Overview

SHINE (SHear INference Environment) is a JAX-powered framework for probabilistic shear estimation in weak gravitational lensing. It treats shear measurement as a Bayesian inverse problem: generating forward models of the sky, convolving with instrument response, and comparing to observed data to infer posterior distributions of shear parameters.

**Status:** Early development / Alpha. Core modules (`shine.config`, `shine.inference`) and the first instrument backend (`shine.euclid`) are implemented. See `DESIGN.md` for the full architecture.

**Organization:** CosmoStat Lab (CEA / CNRS)
**License:** MIT

## Repository Structure

```
SHINE/
├── .github/workflows/       # CI/CD (Claude PR assistant + code review)
├── assets/
│   └── logo.png             # Project logo
├── configs/
│   └── euclid_vis.yaml      # Example Euclid VIS config
├── data/
│   └── EUC_VIS_SWL/         # Euclid VIS test data (Git LFS)
├── external/
│   └── GalSim/              # External GalSim dependency (placeholder)
├── notebooks/
│   └── euclid_vis_map.ipynb  # MAP fitting demo notebook
├── scripts/
│   └── test_map.py          # Standalone MAP test script
├── shine/                   # Main Python package
│   ├── __init__.py
│   ├── config.py            # Base inference configuration
│   ├── inference.py         # Inference engine (MAP, NUTS)
│   └── euclid/              # Euclid VIS instrument backend
│       ├── config.py        # Euclid-specific configuration
│       ├── data_loader.py   # FITS data loading & source selection
│       ├── scene.py         # Multi-exposure scene model (NumPyro)
│       └── plots.py         # Diagnostic visualizations
├── tests/
│   └── test_euclid/         # Euclid module tests (15 tests)
├── CLAUDE.md                # This file
├── DESIGN.md                # Architecture & design document
├── LICENSE                  # MIT License
├── README.md                # Project overview and quick start
└── pyproject.toml           # Build configuration
```

## Key Technologies

- **JAX** — Core computation: JIT compilation, vmap vectorization, grad for HMC
- **NumPyro** — Probabilistic programming: hierarchical models, MCMC (NUTS/HMC)
- **JAX-GalSim** — Differentiable galaxy profile rendering and PSF convolution
- **BlackJAX** — Optional lower-level inference library for custom samplers

## Module Architecture

| Module | Status | Purpose |
|--------|--------|---------|
| `shine.config` | Implemented | Inference configuration (MAP, NUTS, VI settings) |
| `shine.inference` | Implemented | Inference engine (MAP optimization, NUTS via NumPyro) |
| `shine.euclid` | Implemented | Euclid VIS instrument backend: data loading, scene model, diagnostics |
| `shine.scene_modelling` | Planned | Generic NumPyro generative model definitions |
| `shine.simulations` | Planned | Additional survey interfaces (LSST, MeerKAT) |
| `shine.morphology` | Planned | Non-parametric galaxy profiles (VAE/Diffusion) |
| `shine.wms` | Planned | Workflow management for HPC/SLURM clusters |

### `shine.euclid` — Euclid VIS Backend

The first instrument backend, providing end-to-end shear inference on Euclid VIS quadrant-level data:

- **`config.py`** — Pydantic configuration: data paths, source selection (SNR, `det_quality_flag`, size filtering), prior distributions, multi-tier stamp sizes
- **`data_loader.py`** — Reads quadrant FITS files (SCI/RMS/FLG), PSF grids with bilinear interpolation, background maps, MER catalogs; computes per-source WCS positions, Jacobians, PSF stamps, and visibility
- **`scene.py`** — NumPyro generative model: renders Sersic galaxies convolved with spatially-varying PSFs via JAX-GalSim; multi-tier stamp sizes (64/128/256 px) with separate `vmap` per tier; standalone `render_model_images()` for post-inference visualization
- **`plots.py`** — 3-panel diagnostic figures (observed | model | chi residual) with configurable masking

## Build System

- **Build backend:** setuptools (>=61) with setuptools-scm (>=6.2)
- **Version:** Dynamic, managed by setuptools-scm (writes to `shine/_version.py`)
- **Python:** >=3.9 (supports 3.9 through 3.13)
- **Install:** `pip install -e .` for development

## Code Standards (from DESIGN.md Section 4.1)

When implementing code for this project, follow these conventions:

- **Formatter:** Black
- **Import sorting:** isort
- **Type hints:** Full PEP 484 compliance required
- **Docstrings:** Google-style
- **Testing:** pytest + chex (JAX-specific array shape/type testing)
- **Documentation:** Sphinx + ReadTheDocs (not yet configured)

## JAX-Specific Guidelines

- JAX-GalSim objects are **immutable** — the pipeline must be purely functional
- Use `jax.vmap` or `numpyro.plate` for vectorization over galaxies; **never use Python loops** in rendering paths
- Use `jax.jit` to compile likelihood and gradient functions
- JAX uses a **functional PRNG** — manage RNG keys carefully, especially within NumPyro models
- Support reparameterization (e.g., `LocScaleReparam`) for hierarchical models

## Testing

The `tests/test_euclid/` directory contains 15 tests covering:

- **Unit tests:** PSF grid interpolation, exposure reading, WCS transforms, source selection (SNR, flag, size filters), ExposureSet assembly
- **Scene tests:** Single-exposure rendering, multi-exposure model, multi-tier stamp rendering, `render_model_images()`
- **Integration test:** End-to-end MAP inference on real Euclid VIS data

Run tests with:

```bash
pytest tests/test_euclid/ -q
```

Future testing should also include:
- Self-consistency: generate data with known shear, infer it back, verify posterior credible intervals
- Comparison: compare with standard (non-JAX) GalSim for numerical accuracy

## Configuration Pattern

SHINE uses GalSim-compatible YAML configuration with a probabilistic extension: any parameter defined as a distribution (e.g., `type: Normal`) becomes a **latent variable** for inference rather than a fixed simulation value. See `DESIGN.md` Section 6.1 for config examples.

## Development Roadmap

1. **Phase 1:** Prototype with simple parametric models (Sersic) and constant PSF
2. **Phase 2:** Realistic PSF models (Euclid/LSST specific)
3. **Phase 3:** Non-parametric galaxy morphology (VAE/Diffusion)
4. **Phase 4:** Large-scale validation on Flagship/CosmoDC2 simulations

## Key Design Reference

The primary design document is `DESIGN.md` (343 lines). Consult it for:
- Architecture diagrams (Section 2.3)
- Component API designs (Section 3)
- Code structure examples (Section 3.2)
- End-to-end usage examples with config and Python code (Section 6)

## Common Tasks

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest

# Run Euclid tests only
pytest tests/test_euclid/ -q

# Standalone MAP test on bundled data
python scripts/test_map.py

# Format code
black shine/
isort shine/
```

## Notes for AI Assistants

- Always consult `DESIGN.md` before implementing new modules — it contains detailed API specifications and code structure examples
- The `shine.euclid` package is the reference instrument backend — follow its patterns (config, data loader, scene model) when adding new instruments
- Test data for Euclid is stored in `data/EUC_VIS_SWL/` via Git LFS; FITS files are LFS-tracked, README.md and .py scripts are regular git
- The `external/GalSim/` directory is currently an empty placeholder
- CI/CD currently only includes Claude-based PR workflows; traditional CI (tests, linting) should be added
