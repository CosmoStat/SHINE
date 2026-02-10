# CLAUDE.md - AI Assistant Guide for SHINE

## Project Overview

SHINE (SHear INference Environment) is a JAX-powered framework for probabilistic shear estimation in weak gravitational lensing. It treats shear measurement as a Bayesian inverse problem: generating forward models of the sky, convolving with instrument response, and comparing to observed data to infer posterior distributions of shear parameters.

**Status:** Early development / Alpha. Core pipeline (config, data, scene, inference) is implemented. Validation infrastructure through Level 1 (noise bias) is complete.

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
├── configs/
│   └── validation/          # Validation pipeline configs
│       ├── level0_base.yaml # Level 0: noiseless sanity check (SHINE config)
│       ├── level0_test.yaml # Level 0: campaign config
│       ├── level1_base.yaml # Level 1: single galaxy + noise (SHINE config)
│       └── level1_test.yaml # Level 1: campaign config
├── examples/
│   └── shear_inference.py   # Pedagogical walkthrough
├── external/
│   └── GalSim/              # External GalSim dependency (placeholder)
├── shine/                   # Main Python package
│   ├── __init__.py          # Package init
│   ├── _version.py          # Auto-generated version (setuptools-scm)
│   ├── config.py            # YAML config parsing + pydantic models
│   ├── data.py              # DataLoader + synthetic observation generation
│   ├── galaxy_utils.py      # Galaxy profile helpers (Exponential, Sersic)
│   ├── inference.py         # Inference engine (NUTS, MAP, VI)
│   ├── main.py              # CLI entry point for full pipeline
│   ├── psf_utils.py         # PSF construction helpers
│   ├── scene.py             # SceneBuilder (NumPyro model construction)
│   └── validation/          # Bias measurement infrastructure
│       ├── __init__.py      # Public API exports
│       ├── bias_config.py   # Pydantic models for bias testing
│       ├── cli.py           # Three-stage CLI (run / extract / stats)
│       ├── extraction.py    # Posterior → structured results
│       ├── plots.py         # Diagnostic and bias plots
│       ├── simulation.py    # Paired-shear simulation driver
│       └── statistics.py    # Bias regression, coverage, SBC
├── tests/                   # pytest test suite
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_galaxy_utils.py
│   ├── test_psf_utils.py
│   └── test_validation/     # Validation-specific tests
│       ├── test_bias_config.py
│       ├── test_extraction.py
│       ├── test_simulation.py
│       ├── test_statistics.py
│       ├── test_batched_inference.py
│       ├── test_level0_integration.py
│       ├── test_level0_batched_integration.py
│       └── test_level1_integration.py
├── CLAUDE.md                # This file
├── DESIGN.md                # Comprehensive architecture & design document
├── LICENSE                  # MIT License
├── README.md                # Project overview and quick start
└── pyproject.toml           # Build configuration + tool settings
```

## Key Technologies

- **JAX** — Core computation: JIT compilation, vmap vectorization, grad for HMC
- **NumPyro** — Probabilistic programming: hierarchical models, MCMC (NUTS/HMC)
- **JAX-GalSim** — Differentiable galaxy profile rendering and PSF convolution
- **SciPy** — Statistical tests (KS test for SBC, normal quantiles for coverage)
- **ArviZ** — Posterior diagnostics (R-hat, ESS, BFMI)
- **Matplotlib** — Diagnostic and validation plots

## Module Architecture

| Module | Purpose | Status |
|--------|---------|--------|
| `shine.config` | YAML config parsing and validation (pydantic) | Implemented |
| `shine.data` | DataLoader, synthetic observation generation | Implemented |
| `shine.scene` | NumPyro generative model (SceneBuilder) | Implemented |
| `shine.inference` | Bayesian inference (NUTS, MAP, VI) | Implemented |
| `shine.galaxy_utils` | Galaxy profile construction | Implemented |
| `shine.psf_utils` | PSF construction (GalSim + JAX-GalSim) | Implemented |
| `shine.validation` | Bias measurement infrastructure | Implemented (L0 + L1) |
| `shine.morphology` | Non-parametric morphology (VAE/GAN) | Not yet implemented |
| `shine.wms` | Workflow management for HPC/SLURM | Not yet implemented |

## Validation Pipeline

The validation pipeline measures shear estimation bias through a hierarchy of levels:

### Level 0 — Noiseless sanity check
- Near-zero noise, fixed ellipticity
- MAP or MCMC inference
- Checks: posterior collapses on truth, |m| < 1%

### Level 1 — Single galaxy with noise (noise bias)
- Real Gaussian noise, random intrinsic ellipticity
- Paired-shear method (+g/-g cancels shape noise)
- Flux/noise parameter sweep (4 flux x 4 noise grid)
- NUTS inference (500 warmup, 1000 samples, 2 chains)
- Statistics: bias regression, paired response, coverage, SBC ranks
- Acceptance: |m| < 0.01, |c| < 5e-4, coverage within 3%, SBC KS p > 0.01

### Three-stage CLI

```bash
# Stage 1: Generate data + run MCMC
shine-bias-run --shine-config configs/validation/level1_base.yaml \
  --g1-true 0.02 --g2-true 0.0 --seed 42 --paired \
  --flux 1000 --noise-sigma 0.1 --output-dir results/level1

# Stage 2: Extract posteriors → CSV
shine-bias-extract --input-dir results/level1 --output results/level1/summary.csv

# Stage 3: Compute bias statistics + plots
shine-bias-stats --input results/level1/summary.csv --output-dir results/level1/stats \
  --level level_1 --posterior-dir results/level1
```

## Build System

- **Build backend:** setuptools (>=61) with setuptools-scm (>=6.2)
- **Version:** Dynamic, managed by setuptools-scm (writes to `shine/_version.py`)
- **Python:** >=3.11 (supports 3.11 through 3.13)
- **Install:** `pip install -e .` for development
- **Dependencies:** JAX, NumPyro, GalSim, JAX-GalSim, pydantic, PyYAML, ArviZ, matplotlib, SciPy

## Code Standards (from DESIGN.md Section 4.1)

When implementing code for this project, follow these conventions:

- **Formatter:** Black (line-length 88, configured in pyproject.toml)
- **Import sorting:** isort (profile "black", configured in pyproject.toml)
- **Type hints:** Full PEP 484 compliance required
- **Docstrings:** Google-style
- **Testing:** pytest with `slow` and `integration` markers
- **Documentation:** MkDocs Material + mkdocstrings

## JAX-Specific Guidelines

- JAX-GalSim objects are **immutable** — the pipeline must be purely functional
- Use `jax.vmap` or `numpyro.plate` for vectorization over galaxies; **never use Python loops** in rendering paths
- Use `jax.jit` to compile likelihood and gradient functions
- JAX uses a **functional PRNG** — manage RNG keys carefully, especially within NumPyro models
- Support reparameterization (e.g., `LocScaleReparam`) for hierarchical models

## Testing

The test suite covers unit, integration, and validation tests:

```bash
# Run all fast tests
pytest tests/ -m "not slow"

# Run statistics tests only
pytest tests/test_validation/test_statistics.py -v

# Run slow integration tests (requires MCMC)
pytest tests/ -m slow -v

# Run with coverage
pytest tests/ -m "not slow" --cov=shine
```

**Test markers:**
- `slow` — Tests that run MCMC inference (minutes)
- `integration` — End-to-end pipeline tests

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
pip install -e ".[dev,test]"

# Run fast tests
pytest tests/ -m "not slow"

# Run full test suite (including MCMC)
pytest tests/ -v

# Format code
black shine/
isort shine/

# Run a single Level 1 paired realization
shine-bias-run --shine-config configs/validation/level1_base.yaml \
  --g1-true 0.02 --g2-true 0.0 --seed 42 --paired --output-dir /tmp/test
```

## Notes for AI Assistants

- Always consult `DESIGN.md` before implementing new modules — it contains detailed API specifications and code structure examples
- The validation pipeline follows a strict level hierarchy (L0 → L1 → L2 → L3). Each level adds complexity. Check `shine/validation/bias_config.py` for `BiasLevel` enum
- When adding new statistics or plots, add them to `shine/validation/__init__.py` exports and `__all__`
- CLI entry points are defined in `pyproject.toml` under `[project.scripts]`
- Validation configs live in `configs/validation/` — one `*_base.yaml` (SHINE config) and one `*_test.yaml` (campaign config) per level
- The `external/GalSim/` directory is currently an empty placeholder
- CI/CD currently only includes Claude-based PR workflows; traditional CI (tests, linting) should be added
