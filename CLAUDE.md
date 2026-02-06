# CLAUDE.md - AI Assistant Guide for SHINE

## Project Overview

SHINE (SHear INference Environment) is a JAX-powered framework for probabilistic shear estimation in weak gravitational lensing. It treats shear measurement as a Bayesian inverse problem: generating forward models of the sky, convolving with instrument response, and comparing to observed data to infer posterior distributions of shear parameters.

**Status:** Early development / Alpha. The architectural design is complete (see `DESIGN.md`) but source code implementation has not yet begun — currently only `shine/__init__.py` exists as an empty module.

**Organization:** CosmoStat Lab (CEA / CNRS)
**License:** MIT

## Repository Structure

```
SHINE/
├── .github/workflows/       # CI/CD (Claude PR assistant + code review)
│   ├── claude.yml           # Claude PR assistant triggered by @claude mentions
│   └── claude-code-review.yml # Automated code review on PRs
├── assets/
│   └── logo.png             # Project logo
├── external/
│   └── GalSim/              # External GalSim dependency (placeholder)
├── shine/                   # Main Python package
│   └── __init__.py          # Currently empty
├── CLAUDE.md                # This file
├── DESIGN.md                # Comprehensive architecture & design document
├── LICENSE                  # MIT License
├── README.md                # Project overview and quick start
└── pyproject.toml           # Build configuration
```

## Key Technologies

- **JAX** — Core computation: JIT compilation, vmap vectorization, grad for HMC
- **NumPyro** — Probabilistic programming: hierarchical models, MCMC (NUTS/HMC)
- **JAX-GalSim** — Differentiable galaxy profile rendering and PSF convolution
- **BlackJAX** — Optional lower-level inference library for custom samplers

## Planned Module Architecture

These modules are specified in `DESIGN.md` but not yet implemented:

| Module | Purpose |
|--------|---------|
| `shine.config` | YAML config parsing and validation (pydantic) |
| `shine.scene_modelling` | NumPyro generative model definitions |
| `shine.inference` | Bayesian inference (NUTS, SVI, BlackJAX) |
| `shine.simulations` | Survey-specific data interfaces (Euclid, LSST, MeerKAT) |
| `shine.morphology` | Galaxy surface brightness profiles (Sersic, VAE/GAN) |
| `shine.wms` | Workflow management for HPC/SLURM clusters |

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

## Testing Strategy

Not yet implemented. When tests are added, follow this strategy:

- **Unit tests:** Verify individual components (e.g., Sersic profile generation)
- **Integration tests:** End-to-end runs on small synthetic patches
- **Validation tests:**
  - Self-consistency: generate data with known shear, infer it back, verify posterior credible intervals
  - Comparison: compare with standard (non-JAX) GalSim for numerical accuracy
- **Run tests with:** `pytest` (when test infrastructure exists)

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

# Run tests (when available)
pytest

# Format code (when tooling is configured)
black shine/
isort shine/
```

## Notes for AI Assistants

- Always consult `DESIGN.md` before implementing new modules — it contains detailed API specifications and code structure examples
- The project has no runtime dependencies listed in `pyproject.toml` yet; add JAX, NumPyro, JAX-GalSim, etc. when implementing modules
- No linter/formatter configuration files exist yet (no `.flake8`, `pyproject.toml [tool.black]`, etc.) — create them following the standards in DESIGN.md Section 4.1 when needed
- No test directory or test infrastructure exists yet — create `tests/` following pytest conventions when adding tests
- The `external/GalSim/` directory is currently an empty placeholder
- CI/CD currently only includes Claude-based PR workflows; traditional CI (tests, linting) should be added when code exists to test
