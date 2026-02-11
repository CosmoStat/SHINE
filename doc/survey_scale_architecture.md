# SHINE Survey-Scale Architecture

## Part I — Overview

### 1. Introduction

SHINE currently performs shear inference on a single Euclid VIS quadrant
(~600 sources, 3 exposures) using a single consumer-grade GPU with 10 GB
of VRAM.  Scaling to the full Euclid Wide Survey requires processing
~15,000 deg² of sky containing billions of sources observed across
~50,000 pointings, each with 36 CCDs × 4 quadrants and ~4 dithered
exposures.

This document specifies the architecture that bridges the gap between
the single-quadrant prototype and survey-scale production.  It is
organized in four parts:

- **Part I** — The problem, design principles, spatial decomposition,
  and the two-project split.
- **Part II** — Changes to the SHINE library to process one sub-tile.
- **Part III** — The `shine-pipeline` project that orchestrates
  survey-scale inference.
- **Part IV** — Deployment, cost estimates, and phased adoption.

### 2. Design Principles

1. **Source-centric, not detector-centric.**  The processing unit is a
   region of the sky (a sub-tile), not a CCD quadrant.  Each sub-tile
   collects all quadrant exposures that overlap its footprint, regardless
   of which CCD or pointing they come from.

2. **Adopt Euclid's own spatial decomposition.**  The Euclid Science
   Ground Segment (SGS) already partitions the sky into MER tiles
   (32' × 32').  SHINE reuses this tiling rather than inventing its own.

3. **Quadrants are immutable data.**  Quadrant images (science, RMS,
   flags) are loaded exactly as delivered by the VIS pipeline — no
   trimming, no reprojection, no pixel modification.  The forward model
   renders galaxies in each quadrant's native pixel coordinate system
   using its own WCS and PSF.

4. **Embarrassingly parallel.**  Sub-tiles are fully independent
   inference problems with no cross-sub-tile communication.

5. **Separate the model from the pipeline.**  SHINE (this repo) owns
   the probabilistic model, rendering, and inference.  A separate
   project (`shine-pipeline`) owns tiling, data preparation,
   orchestration, and catalog assembly.  The contract between them is a
   manifest file.

6. **Cloud-native, SLURM-compatible.**  The production stack uses modern
   cloud tooling (Flyte, SkyPilot, Iceberg), but the architecture
   degrades gracefully to a SLURM cluster since each sub-tile is just a
   Python function call.

### 3. Spatial Hierarchy

#### 3.1 MER Tiles (Data Preparation Unit)

The Euclid MER pipeline partitions the sky into tiles defined in the
Euclid Data Product Description Document (DPDD):

| Property | Value |
|----------|-------|
| Extended area | 32' × 32' (object detection & measurement) |
| Core area | 30' × 30' (catalog output, no overlap) |
| Overlap | 2' border with adjacent tiles |
| Grid spacing | 30' between tile centers |
| Identification | 9-digit unique integer (e.g., `102159490`) |
| Coordinate system | Tangential projection at tile center |

The MER tile serves as the **data preparation boundary**: given a tile
ID, the pipeline discovers all VIS quadrant exposures that overlap the
tile footprint and loads the MER source catalog.

A MER tile is too large for a single GPU.  At ~475,000 detected sources
per deg² (~30,000–50,000 after source selection per tile), the memory
requirements exceed available VRAM.

#### 3.2 Sub-Tiles (Inference Unit)

Each MER tile is subdivided into a regular grid of **sub-tiles**.  The
sub-tile is the fundamental inference unit: one sub-tile = one GPU job.

**Baseline configuration: 4 × 4 grid** within each MER tile.

| Property | Value |
|----------|-------|
| Core area | 8' × 8' (4,800 × 4,800 px at 0.1"/px) |
| Extended area | 10' × 10' (6,000 × 6,000 px at 0.1"/px) |
| Overlap margin | 1' per side (600 px) |
| Sources (after selection) | ~2,000–3,000 per sub-tile |
| Overlapping quadrants | ~4–8 (dithered pointings × overlapping CCDs) |
| Shear model | Constant (g1, g2) per sub-tile |
| GPU memory (estimated) | 2–4 GB |

The sub-tile grid is configurable.  The sub-tile side length trades off
GPU memory against the number of sources constraining the shear:

| Sub-tile side | Grid | Sources | Shear precision (σ_g/√N) | GPU fit? |
|---------------|------|---------|--------------------------|----------|
| 4' | 8 × 8 | ~500 | ~0.002 | easily |
| 8' | 4 × 4 | ~2,000 | ~0.001 | comfortable |
| 16' | 2 × 2 | ~7,500 | ~0.0006 | tight |
| 32' | 1 × 1 | ~30,000 | ~0.0003 | does not fit |

At 8' per sub-tile the shear field is well-approximated as constant
(the coherence scale of cosmic shear is ~arcminutes), and the
per-sub-tile source count delivers shear precision well below the
shape-noise floor.

#### 3.3 Hierarchy Diagram

```
Euclid Wide Survey (15,000 deg²)
 │
 ├── MER Tile 102159490  (32' × 32', data preparation unit)
 │    │
 │    ├── Sub-tile (0,0)  (8'+2×1' extended, inference unit → 1 GPU)
 │    │    ├── ~2,500 sources (in extended area)
 │    │    ├── ~6 full quadrant images (from 4 pointings, each 2048×2066)
 │    │    ├── local (g1, g2) estimation
 │    │    └── report results for core area only
 │    │
 │    ├── Sub-tile (0,1) ...
 │    ├── ...
 │    └── Sub-tile (3,3)
 │
 ├── MER Tile 102159491 ...
 └── ...

Total: ~60,000 MER tiles × 16 sub-tiles = ~960,000 independent GPU jobs
```

### 4. Boundary Handling

#### 4.1 The Problem

A source near a sub-tile boundary has a rendering stamp that extends
beyond the sub-tile's core area.  If the neighboring sub-tile does not
model this source, it will see unexplained flux near the boundary,
potentially biasing the shear estimate.  Similarly, blended source
groups that straddle a boundary must be modeled jointly by at least one
sub-tile.

#### 4.2 Solution: Core / Extended Areas

Following the same pattern used by the Euclid MER pipeline at the tile
level, each sub-tile defines two concentric areas:

```
┌───────────────────────────┐
│  Extended area (10' × 10') │
│  ┌─────────────────────┐  │
│  │ Core area (8' × 8')  │  │
│  │                     │  │
│  │  Report shear and   │  │
│  │  source params here │  │
│  │                     │  │
│  └─────────────────────┘  │
│  ← 1' overlap margin →    │
│  Model these sources but   │
│  discard their results     │
└───────────────────────────┘
```

Sources in the **extended area** are modeled during inference; results
are reported only for **core area** sources.  Overlap-margin sources
are modeled by adjacent sub-tiles independently and serve only to
ensure the scene model is complete near the boundary.

#### 4.3 Margin Sizing

The overlap margin must be large enough that no source in the core area
has its rendering stamp overlapping with an unmodeled source outside the
extended area:

```
margin ≥ max_stamp_size / 2  (in sky coordinates)
256 / 2 × 0.1"/px = 12.8" ≈ 0.21'
```

A 1' (60") margin provides ~4.7× the minimum required clearance,
accommodating even unusually extended sources and ensuring blend groups
near the boundary are fully contained.

#### 4.4 Overlap Overhead

The extended area is (10/8)² = 1.5625× the core area, meaning each
sub-tile models ~56% more sources than strictly needed.  In practice
the overhead is lower because overlap sources tend to be at the smallest
stamp tier and rendering cost is dominated by the largest-stamp sources
deep in the core.

#### 4.5 Inter-Tile Boundaries

The same core/extended pattern applies at the MER tile level.  When
processing sub-tiles near the edge of a MER tile, the sub-tile's
extended area may extend beyond the MER tile boundary into the adjacent
tile's overlap region.  The 2' MER tile overlap (≥ the 1' sub-tile
margin) ensures that source and image data is always available.

### 5. Two-Project Architecture

The work is split across two repositories with a clear contract between
them:

```
┌─────────────────────────────────────────────────────────────┐
│  shine-pipeline  (new repo)                                  │
│                                                              │
│  Tiling · Quadrant discovery · Source partitioning            │
│  Manifest generation · Flyte workflows · SkyPilot compute    │
│  Iceberg catalog assembly · Terraform infrastructure         │
│                                                              │
│  Produces:  per-sub-tile manifest (YAML)                     │
│  Consumes:  per-sub-tile results (Parquet)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │  manifest.yaml
                           │  (quadrant paths + source list + sky bounds)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  shine  (this repo)                                          │
│                                                              │
│  Galaxy model · Rendering · Likelihood · Inference           │
│  Data loading · PSF interpolation · Diagnostics              │
│                                                              │
│  Consumes:  manifest (list of quadrants + sources)           │
│  Produces:  shear + per-source params (Parquet)              │
└─────────────────────────────────────────────────────────────┘
```

#### 5.1 Why This Split

| Concern | SHINE | shine-pipeline |
|---------|-------|----------------|
| Galaxy modeling & rendering | **Yes** | No |
| Inference (MAP / VI / NUTS) | **Yes** | No (calls SHINE) |
| Knows about FITS data formats | **Yes** | Only file paths |
| Knows about tiles / sub-tiles | No | **Yes** |
| Knows about Flyte / SkyPilot | No | **Yes** |
| Knows about Iceberg / S3 tiers | No | **Yes** |
| Terraform / Docker / CI | No | **Yes** |
| Can run on a laptop | **Yes** | Orchestrates cloud |
| Installable as a library | `pip install shine` | `pip install shine-pipeline` |

A scientist doing methods development works in SHINE with a local FITS
file.  The pipeline team wraps it for survey-scale production.  Changes
to the probabilistic model don't touch the pipeline; changes to
infrastructure don't touch the model.

#### 5.2 The Manifest Contract

The manifest is the interface between the two projects.  It is produced
by `shine-pipeline` and consumed by SHINE:

```yaml
# subtile_manifest.yaml
tile_id: 102159490
sub_tile_row: 2
sub_tile_col: 1

sky_bounds:
  core_ra: [269.123, 269.256]       # degrees
  core_dec: [-28.456, -28.323]
  extended_ra: [269.106, 269.273]
  extended_dec: [-28.473, -28.306]

quadrants:
  - sci_path: s3://euclid-q1/VIS/quad_3-4-F_exp0.fits.gz
    rms_path: s3://euclid-q1/VIS/rms_3-4-F_exp0.fits.gz
    flg_path: s3://euclid-q1/VIS/flg_3-4-F_exp0.fits.gz
    psf_path: s3://euclid-q1/VIS/PSF_3-4-F.fits.gz
    bkg_path: s3://euclid-q1/VIS/bkg_3-4-F_exp0.fits.gz
  - sci_path: s3://euclid-q1/VIS/quad_4-4-E_exp0.fits.gz
    # ...

source_catalog: s3://euclid-q1/MER/catalog_102159490.fits.gz
source_ids: [1042, 1043, 1044, ...]   # extended-area sources
core_source_ids: [1042, 1043, ...]     # core-area sources (results reported)
```

SHINE reads this manifest, loads the referenced data, builds an
`ExposureSet`, runs inference, and writes results.  SHINE does not need
to know how this manifest was produced.

---

## Part II — SHINE (The Inference Library)

### 6. What SHINE Does Today

SHINE's Euclid backend processes a single CCD quadrant across dithered
pointings:

1. **Data loading** (`data_loader.py`): reads quadrant FITS files
   (science, RMS, flags), interpolates PSFs from a grid, projects
   catalog sources onto pixel coordinates via WCS, computes Jacobians
   and visibility.  Produces an `ExposureSet` — stacked arrays of shape
   `(n_exp, ny, nx)` for images and `(n_sources, n_exp, ...)` for
   per-source metadata.

2. **Scene modeling** (`scene.py`): a NumPyro generative model.  For
   each source, renders a galaxy profile (via JAX-GalSim), convolves
   with the interpolated PSF, and scatter-adds onto the model image.
   Sources are processed in stamp-size tiers (64/128/256 px) with
   separate `vmap` per tier.

3. **Inference** (`inference.py`): runs MAP (SVI with AutoDelta),
   VI (SVI with AutoNormal), or NUTS (full MCMC) on the scene model.

4. **Diagnostics** (`plots.py`): 3-panel figures (observed | model |
   chi residual).

### 7. Changes for Sub-Tile Processing

The core rendering and inference code remain unchanged.  The changes
are in data loading and source filtering.

#### 7.1 Multi-CCD ExposureSet

**Current state:** All "exposures" in an `ExposureSet` come from the
same CCD quadrant across dithered pointings.  They share the same
quadrant ID, pixel dimensions (2048 × 2066), and PSF grid.

**Sub-tile state:** Exposures come from multiple CCDs across multiple
pointings.  Each quadrant has its own WCS, PSF grid, and detector
characteristics — but they all share the same 2048 × 2066 pixel
dimensions.

| | Current | Sub-tile |
|---|---|---|
| What is an exposure? | A quadrant from one dithered pointing | A quadrant from one CCD of one pointing |
| How many per job? | 3–4 (same CCD, dithered) | ~4–8 (dithered pointings × overlapping CCDs) |
| Image shape | Uniform (2048 × 2066) | Uniform (2048 × 2066) |
| Same CCD across exposures? | Yes | No |
| PSF model per exposure | Single grid | Different grid per quadrant |

Since all VIS quadrants share the same pixel dimensions, they stack
directly into the existing `ExposureSet` layout `(n_exp, 2066, 2048)`
with no padding or reshaping.  The `ExposureSet` dataclass works
without any modification.

The only change is in `data_loader.py`: generalize `EuclidDataLoader`
to accept quadrants from multiple CCDs, each with its own PSF grid and
WCS.  The per-source metadata computation
(`_compute_source_metadata`) already handles per-exposure WCS and PSF —
it just needs to accept different grids per exposure.

#### 7.2 Rendering in Native Quadrant Coordinates

The forward model renders each source into each quadrant using that
quadrant's own coordinate system.  For a given source at sky position
(RA, Dec) and a given quadrant:

1. The source's pixel position `(x, y)` is computed via the quadrant's
   WCS (`all_world2pix`).
2. The local WCS Jacobian `(dudx, dudy, dvdx, dvdy)` is evaluated at
   that pixel position.
3. The PSF is interpolated from the quadrant's PSF grid at that pixel
   position.
4. The galaxy is rendered on a stamp centered at `(x, y)` using the
   local WCS and interpolated PSF.
5. The stamp is scatter-added onto the model image at the source's
   pixel position.

This is exactly what the existing `_render_tier` and
`_compute_source_metadata` already do.  No coordinate transformation
is needed because each quadrant is processed in its own native frame.

#### 7.3 Unmodeled Sources Outside the Sub-Tile

A quadrant may contain sources that fall outside the sub-tile's
extended area.  These sources contribute flux to the observed image but
are not included in the model.

This does **not** bias the inference.  The likelihood gradient with
respect to any model parameter θ at a pixel p where the model has zero
flux is:

```
∂L/∂θ |_p  =  (obs[p] − model[p]) / σ[p]²  ×  ∂model[p]/∂θ  =  0
```

because `∂model[p]/∂θ = 0` at pixels where no modeled source has a
stamp.  Unmodeled sources create squared residuals in the loss value
but contribute exactly zero gradient, so they affect neither MAP
estimates nor MCMC posterior geometry.

#### 7.4 Visibility-Aware Source Filtering

**Current state.**  `_render_tier` vmaps `render_one_galaxy` over all
sources in a stamp-size tier for every exposure, using `source_visible`
to mask invisible sources.  Masked sources still execute the full FFT
convolution pipeline (with dummy parameters), wasting computation.

At single-quadrant scale (600 sources × 3 exposures) this overhead is
negligible.  At sub-tile scale (~2,500 sources × ~6 quadrants), each
quadrant only sees ~400–800 of the 2,500 sources.  Without filtering,
the vmap would execute ~15,000 renders of which only ~3,000–5,000 are
real — a ~3–5× waste.

**Proposed change.**  Replace the current `_compute_tier_indices`
(which partitions sources by stamp tier only) with a two-level
partitioning by **(tier, quadrant)**.  This is computed once during
`ExposureSet` assembly, before any JIT compilation:

```python
# During ExposureSet assembly (data preparation, not inference):
tier_quad_indices = []
for t in range(n_tiers):
    tier_sources = np.where(source_stamp_tier == t)[0]
    per_quad = []
    for j in range(n_exp):
        visible = source_visible[tier_sources, j]
        per_quad.append(jnp.array(tier_sources[visible], dtype=jnp.int32))
    tier_quad_indices.append(per_quad)
```

Then in `_render_tier`, use the precomputed index array:

```python
indices = tier_quad_indices[tier_idx][exp_idx]
if indices.shape[0] == 0:
    return model_image

# vmap over only the visible subset
flux_t = flux[indices]
hlr_t = hlr[indices]
# ...
```

The index arrays are static constants with shapes fixed at data
preparation time — nothing dynamic during inference, no impact on JIT
compilation.

### 8. SHINE Module Structure

```
shine/
├── config.py                # Galaxy model, distribution configs (unchanged)
├── prior_utils.py           # Config → NumPyro sample sites (unchanged)
├── inference.py             # MAP / VI / NUTS engine (unchanged)
└── euclid/
    ├── config.py            # EuclidInferenceConfig (unchanged)
    ├── data_loader.py       # Generalized: multi-CCD ExposureSet,
    │                        #   visibility-aware tier indices
    ├── scene.py             # Visibility filtering in _render_tier
    ├── plots.py             # Diagnostics (unchanged)
    ├── catalog.py           # NEW: read MER catalog, source selection
    └── cli.py               # NEW: manifest-driven entry point
```

#### 8.1 New: `catalog.py`

Utility module that knows how to read a MER source catalog (from local
path or S3), apply SHINE's source selection filters (SNR, flags, size,
point-source exclusion), and return a clean source table.  This
encapsulates Euclid-specific catalog knowledge within SHINE.

#### 8.2 New: `cli.py`

Command-line entry point for processing a single sub-tile:

```bash
python -m shine.euclid.run \
    --manifest subtile_manifest.yaml \
    --config inference_config.yaml \
    --output results/
```

Reads the manifest, loads quadrant data, builds an `ExposureSet`, runs
inference, writes results as Parquet.  This is the function that
`shine-pipeline` invokes — whether via Flyte, SLURM, or a direct
Python call.

The same function is also callable as a Python API:

```python
from shine.euclid.cli import run_subtile
result = run_subtile(manifest_path, config_path, output_dir)
```

### 9. SHINE Implementation Roadmap

#### Phase 1: Multi-CCD Data Loader

- Generalize `EuclidDataLoader` to accept quadrants from multiple CCDs,
  each with its own PSF grid and WCS.
- Implement `catalog.py`: MER catalog reading and source selection.
- Implement `cli.py`: manifest-driven entry point.
- **Validation**: verify that a sub-tile `ExposureSet` built from
  multiple CCD quadrants produces correct MAP results when applied to
  the existing test data region (same quadrant loaded as one of several
  "exposures" should reproduce the single-quadrant result).

#### Phase 2: Visibility Filtering

- Add two-level (tier, quadrant) source filtering to `_render_tier` in
  `scene.py`.
- Benchmark rendering time with and without filtering at sub-tile scale.
- **Validation**: confirm that filtered and unfiltered rendering produce
  identical model images (within floating-point tolerance).

---

## Part III — shine-pipeline (The Orchestration Layer)

### 10. Pipeline Overview

`shine-pipeline` is a separate Python package that orchestrates
survey-scale shear inference.  It depends on SHINE as a library
(`pip install shine`) and handles everything outside the GPU inference
box: tiling the sky, discovering data, generating manifests, scheduling
GPU jobs, and assembling the output catalog.

```
Stage 1: Tile Preparation (CPU, I/O-bound)
──────────────────────────────────────────
  For each MER tile:
  1. Query archive for overlapping VIS quadrant exposures
  2. Load MER source catalog
  3. Partition sources into sub-tiles (core + extended)
  4. For each sub-tile, record which quadrant FITS files overlap
     its extended area
  5. Write per-sub-tile manifests to storage

Stage 2: Sub-Tile Inference (GPU, compute-bound)
─────────────────────────────────────────────────
  For each sub-tile (1 GPU per sub-tile):
  1. Invoke SHINE: python -m shine.euclid.run --manifest ...
  2. SHINE loads quadrant images, builds ExposureSet, runs inference
  3. SHINE writes per-sub-tile results (Parquet)

Stage 3: Catalog Assembly (CPU, I/O-bound)
──────────────────────────────────────────
  1. Collect per-sub-tile results
  2. Filter to core-area sources only
  3. Compact into Iceberg shear catalog
  4. Run quality checks (convergence, outliers)
```

### 11. Data Preparation

#### 11.1 Quadrant Discovery

Given a sub-tile footprint defined by its sky bounding box (RA_min,
RA_max, Dec_min, Dec_max), the pipeline identifies all quadrant FITS
files whose detector footprint overlaps this bounding box.  The overlap
test uses the quadrant's WCS to project its four corners onto the sky
and check intersection with the sub-tile's extended area.

#### 11.2 Source Partitioning

Sources from the MER catalog are assigned to sub-tiles based on their
sky coordinates (RA, Dec):

- A source belongs to a sub-tile's **core** if it falls within the
  8' × 8' core area.
- A source belongs to a sub-tile's **extended** area if it falls within
  the 10' × 10' region (core + 1' margin on each side).
- Sources in the overlap margin are assigned to the extended area of
  **all** adjacent sub-tiles that contain them.

The manifest records both the extended source list (for inference) and
the core source list (for result reporting).

#### 11.3 Manifest Generation

For each sub-tile, the pipeline writes a manifest YAML file listing
the quadrant paths, source IDs, and sky bounds (Section 5.2).  These
manifests are stored on S3 (or local disk for SLURM) and passed to
SHINE as input.

### 12. Workflow Orchestration: Flyte

#### 12.1 Why Flyte

Among Python-native workflow orchestrators (Prefect, Dagster, Airflow,
Argo Workflows), Flyte is the only one that simultaneously provides:

- **Map tasks** — a first-class primitive for embarrassingly parallel
  work with metadata-efficient bitset compression (no per-task DAG
  node overhead).
- **First-class GPU requests** — `Resources(gpu="1")` in the task
  decorator; no YAML pod spec manipulation.
- **First-class spot support** — `interruptible=True` enables separate
  system-retry and user-retry budgets.  After exhausting system
  retries, the final attempt automatically falls back to on-demand.
- **Open source** — Apache 2.0.  No per-task pricing (unlike Dagster
  at ~$0.03/credit, which would cost $30–40K for 1M jobs).
- **Proven at scale** — Woven by Toyota, Spotify, Lyft.

**Why not the others:**

| Tool | Disqualifying issue |
|------|---------------------|
| Dagster | Per-credit pricing: ~$30–40K orchestration cost for 1M jobs |
| Airflow | Cannot scale beyond ~1K parallel tasks; scheduler bottleneck |
| Prefect | No map task primitive; weak spot retry; scheduler bottleneck |
| Argo Workflows | YAML-centric; K8s etcd 1.5 MB object limit at 1M nodes |

#### 12.2 Pipeline Definition

```python
from flytekit import task, workflow, Resources, map_task

@task(requests=Resources(cpu="4", mem="16Gi"))
def prepare_tile(tile_id: int) -> list[str]:
    """Write per-sub-tile manifests and return their S3 paths."""
    ...

@task(
    requests=Resources(gpu="1", mem="8Gi"),
    interruptible=True,
    retries=2,
)
def infer_subtile(manifest_path: str) -> str:
    """Invoke SHINE on one sub-tile. Returns result path."""
    from shine.euclid.cli import run_subtile
    return run_subtile(manifest_path)

@task(requests=Resources(cpu="8", mem="32Gi"))
def assemble_catalog(result_paths: list[str]) -> str:
    """Merge sub-tile results into Iceberg shear catalog."""
    ...

@workflow
def shear_pipeline(tile_ids: list[int]) -> str:
    manifests = map_task(prepare_tile)(tile_id=tile_ids)
    all_manifests = flatten(manifests)
    results = map_task(infer_subtile, concurrency=5000)(
        manifest_path=all_manifests
    )
    return assemble_catalog(result_paths=results)
```

At ~960K sub-tiles, the map task fan-out is batched into groups of
5,000–10,000 to stay within FlytePropeller's optimal range.  Nested
parallelism handles the outer loop over batches.

#### 12.3 SLURM Fallback

The same pipeline works on a SLURM cluster without Flyte.  Each
sub-tile is a self-contained job:

```bash
#!/bin/bash
#SBATCH --job-name=shine-tile-102159490
#SBATCH --array=0-15            # 16 sub-tiles per MER tile
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

TILE_ID=102159490
SUB_ROW=$((SLURM_ARRAY_TASK_ID / 4))
SUB_COL=$((SLURM_ARRAY_TASK_ID % 4))

python -m shine.euclid.run \
    --manifest manifests/${TILE_ID}_${SUB_ROW}_${SUB_COL}.yaml \
    --config configs/inference.yaml \
    --output results/${TILE_ID}/
```

This works because SHINE's CLI is orchestrator-agnostic — it takes a
manifest and writes results.  The orchestrator (Flyte, SLURM, or a
shell script) just decides how to schedule the calls.

### 13. GPU Compute

#### 13.1 SkyPilot (Production Campaigns)

SkyPilot is an open-source (BSD) multi-cloud GPU broker from
UC Berkeley.  It provisions spot instances across 20+ clouds,
automatically failing over when capacity is unavailable or instances
are preempted.

- **Cheapest compute.**  Shops across GCP (L4 spot at $0.21/hr), AWS
  (L4 spot at $0.35/hr), RunPod ($0.39/hr), and Vast.ai ($0.20–0.50)
  to find the lowest price.  Blended spot rate: ~$0.25/hr.
- **Automatic spot recovery.**  Preempted jobs are re-provisioned in a
  different region or cloud.
- **Managed Jobs.**  Up to 2,000 concurrent running jobs per controller.
- **Open source.**  No vendor lock-in.

SkyPilot integrates with Flyte as the compute backend: Flyte submits
tasks, SkyPilot provisions the cheapest available GPU instance.

#### 13.2 Modal (Development & Prototyping)

Modal is a serverless GPU platform with a Python-native SDK.  It
eliminates all infrastructure management at the cost of higher per-hour
pricing (~$0.80/hr for an L4).

- **2–5 second cold starts** with GPU memory snapshots.
- **Zero infrastructure.**  No Dockerfiles, no Terraform, no Kubernetes.
- **True pay-per-second** with zero idle cost.
- **`.map()` built-in** — `function.map(manifests)` fans out to
  thousands of GPUs.

```python
import modal

app = modal.App("shine-dev")
image = (modal.Image.debian_slim()
    .pip_install("jax[cuda12]", "numpyro", "jax-galsim", "astropy"))

@app.function(gpu="L4", image=image, timeout=3600)
def infer_subtile(manifest_path: str):
    from shine.euclid.cli import run_subtile
    return run_subtile(manifest_path)

@app.local_entrypoint()
def main():
    manifests = [...]  # list of S3 paths
    results = list(infer_subtile.map(manifests))
```

Use Modal for pilot runs (100–1,000 sub-tiles) and rapid
experimentation.  Switch to SkyPilot for production campaigns where
the 2–3× cost premium adds up.

#### 13.3 Compute Platform Decision Matrix

| Criterion | SkyPilot | Modal | AWS Batch | SLURM |
|-----------|----------|-------|-----------|-------|
| Cost (L4/hr) | ~$0.25 (spot) | ~$0.80 | ~$0.35 (spot) | Allocation-based |
| 200K GPU-hr total | ~$50K | ~$160K | ~$70K | Depends on center |
| Cold start | 2–5 min | 2–5 sec | 2–5 min | ~0 (dedicated) |
| Max concurrency | 2,000 | Enterprise | 10,000+ | Cluster-limited |
| Spot recovery | Cross-cloud | N/A (serverless) | Same-region | N/A |
| Infrastructure | Controller VM | None | Compute env | Cluster |
| Best for | Production | Dev / prototyping | AWS-native | HPC centers |

### 14. Data Lakehouse: Apache Iceberg on S3

#### 14.1 Why Iceberg

The output shear catalog should not remain in FITS.  Apache Iceberg
provides ACID transactions, schema evolution, time travel, and fast
analytical queries — capabilities FITS lacks entirely.  AWS, Google,
Snowflake, and Databricks have all converged on Iceberg as the standard
open table format.

| Feature | Benefit |
|---------|---------|
| **Schema evolution** | Add columns (e.g., VAE morphology params) without rewriting data |
| **Partition evolution** | Change partitioning (e.g., tile_id → HEALPix) without rewriting |
| **Time travel** | Query at any historical snapshot |
| **Row lineage (V3)** | Every row tracks which write operation created it |
| **Compaction** | Merge millions of small per-job Parquet files into optimal sizes |
| **Hidden partitioning** | Partition by `tile_id` transparently to queries |

#### 14.2 Storage Tiers

| Data | Volume | Tier | Rationale |
|------|--------|------|-----------|
| Input FITS (current batch) | ~1 GB/sub-tile | S3 Standard | Read once per job |
| Hot compute staging | ~10 MB/job output | S3 Express One Zone | Same-AZ, 10× lower latency |
| Shear catalog (Iceberg) | ~10 GB Q1, ~TB survey | S3 Standard | Frequent analytical queries |
| Processed input archive | ~30 TB Q1, ~PB survey | S3 Intelligent-Tiering | Auto-archives, zero retrieval fees |

#### 14.3 Output Format: Parquet

Per-job results are written as Parquet and committed to an Iceberg
table.  Parquet provides columnar storage with predicate pushdown,
compression (3–5×), and native support in every modern query engine.
FITS remains for input data only (as delivered by the Euclid pipeline).

SHINE writes Parquet; `shine-pipeline` commits it to Iceberg.

#### 14.4 Metadata Catalog and Query Engines

**AWS Glue Data Catalog** — serverless, integrates natively with Athena
and DuckDB.  For published releases, **Project Nessie** adds Git-like
branching and tagging (e.g., `q1-release-v1.0`).

| Engine | Use case | Deployment |
|--------|----------|------------|
| **DuckDB** | Interactive analysis, notebooks | Embedded — `pip install duckdb` |
| **Polars** | Python DataFrame operations | `pip install polars` — 5–10× faster than Pandas |
| **Athena** | Ad-hoc SQL over full survey | Serverless — $5/TB scanned |

```python
import duckdb

con = duckdb.connect()
con.sql("INSTALL iceberg; LOAD iceberg;")

result = con.sql("""
    SELECT tile_id,
           AVG(g1) AS mean_g1, AVG(g2) AS mean_g2,
           STDDEV(g1) AS std_g1, COUNT(*) AS n_sources
    FROM iceberg_scan('s3://shine-catalog/shear_catalog')
    WHERE snr > 20
    GROUP BY tile_id
""").df()
```

#### 14.5 Shear Catalog Schema

The assembled Iceberg table contains:

| Column | Type | Description |
|--------|------|-------------|
| `TILE_ID` | int64 | MER tile identifier |
| `SUBTILE_ROW` | int8 | Sub-tile row index |
| `SUBTILE_COL` | int8 | Sub-tile column index |
| `RA` | float64 | Sub-tile center RA (deg) |
| `DEC` | float64 | Sub-tile center Dec (deg) |
| `G1` | float32 | Shear component 1 (MAP or posterior mean) |
| `G2` | float32 | Shear component 2 |
| `G1_ERR` | float32 | Uncertainty on g1 |
| `G2_ERR` | float32 | Uncertainty on g2 |
| `N_SOURCES` | int32 | Number of core sources used |
| `METHOD` | string | Inference method (map / vi / nuts) |
| `CONVERGENCE` | float32 | SVI final loss or NUTS r_hat |

Per-source parameters (flux, half-light radius, ellipticity, position
offsets) are stored in a separate Iceberg table partitioned by tile.

### 15. Observability

Flyte provides the primary observability layer.  Grafana adds
infrastructure metrics that Flyte does not own.  Scientific results
live in the Iceberg catalog, not in the observability stack.

#### 15.1 Job State and Logs: Flyte (Built-In)

Flyte's metadata store (FlyteAdmin + PostgreSQL) tracks every task
execution: status, start/end time, duration, retry count, error
messages, spot preemption recoveries.

| Flyte feature | What it provides |
|---------------|------------------|
| **FlyteAdmin DB** | Status of all 1M tasks — queryable via API |
| **FlyteConsole** | Web UI: per-task logs, map task progress (e.g., "847,231 / 960,000"), input/output inspection |
| **Flyte Deck** | Inline HTML reports attached to each task (convergence plots, residual images) |
| **System vs user retries** | Separate tracking of spot interruptions vs application failures |

Flyte Deck is particularly useful — each `infer_subtile` task can
render its observed/model/residual panels directly into FlyteConsole:

```python
import flytekit

@task(requests=Resources(gpu="1"), interruptible=True, retries=2)
def infer_subtile(manifest_path: str) -> str:
    result = run_subtile(manifest_path)
    flytekit.Deck("diagnostics", make_diagnostic_html(result))
    return result.output_path
```

#### 15.2 Infrastructure Metrics: Grafana + Prometheus

| Source | Metrics |
|--------|---------|
| **FlytePropeller** | `flyte_task_count_by_phase`, workflow durations, queue depths |
| **NVIDIA GPU Operator** | SM utilization, memory usage, temperature |
| **CloudWatch / GCP** | Spot interruptions, instance counts, cost |
| **Loki** | Centralized logs (~20 GB for 1M jobs, compressed to ~2 GB on S3) |

All metrics feed into a single Grafana dashboard — one pane of glass
for both workflow health and infrastructure health.

**Cost:** ~$30/month (a `t3.medium` running the Grafana stack).

### 16. Infrastructure as Code: Terraform

All `shine-pipeline` infrastructure is defined in Terraform:

```
shine-pipeline/
├── infra/
│   └── terraform/
│       ├── modules/
│       │   ├── networking/      # VPC, subnets, security groups
│       │   ├── eks/             # EKS cluster for Flyte
│       │   ├── gpu-nodes/       # Karpenter GPU node pools (spot + on-demand)
│       │   ├── storage/         # S3 buckets (input, staging, catalog)
│       │   ├── flyte/           # Flyte control plane (FlyteAdmin, DB, console)
│       │   └── monitoring/      # Grafana, Prometheus, Loki
│       ├── environments/
│       │   ├── dev/             # Small cluster for testing
│       │   └── prod/            # Full-scale survey processing
│       ├── main.tf
│       └── variables.tf
```

### 17. CI/CD and Reproducibility

#### 17.1 Container Pipeline

```
git push → GitHub Actions → Build Docker image (JAX + CUDA + SHINE)
         → Push to ECR (tagged with git SHA)
         → Push to GHCR (public mirror for collaborators)
```

Images are tagged with the git short SHA (`shine:fdef029`) for exact
reproducibility.  Semantic version tags (`shine:v0.3.1`) for releases.

#### 17.2 Tiered Testing

| Trigger | Test suite | GPU? | Cost |
|---------|-----------|------|------|
| Every push | `pytest tests/ -k "not gpu"` | No | Free |
| PR to main | GPU smoke test (single sub-tile MAP) | Yes | ~$0.50 |
| Weekly | Full integration on bundled data | Yes | ~$2 |

#### 17.3 Data Versioning: DVC

DVC stores content-addressable hashes of input FITS files in Git while
actual data lives on S3.  The `dvc.lock` file records the exact hash
of every input, enabling full reconstruction of past runs.

#### 17.4 Provenance Chain

Every output catalog row is traceable:

```
Shear catalog row (Iceberg)
  → Iceberg V3 row lineage (which commit wrote it)
    → FlyteAdmin execution record (tile_id, code_version, config_hash)
      → Git SHA (exact source code)
      → DVC lock (exact input data hashes)
      → ECR image tag (exact Python/JAX/CUDA environment)
```

### 18. shine-pipeline Module Structure

```
shine-pipeline/
├── pipeline/
│   ├── __init__.py
│   ├── tiling.py              # MER tile footprints, sub-tile grid
│   ├── discovery.py           # Query archive / S3 for overlapping quadrants
│   ├── manifest.py            # Write per-sub-tile manifest YAML
│   ├── workflows.py           # Flyte workflow definitions
│   ├── catalog_assembler.py   # Merge results → Iceberg table
│   └── config.py              # Pipeline-level Pydantic config
├── infra/
│   ├── terraform/             # All infrastructure definitions
│   └── docker/                # Dockerfile (JAX + CUDA + SHINE)
├── configs/
│   ├── q1_campaign.yaml       # Q1 pipeline config
│   └── wide_survey.yaml       # Full survey pipeline config
├── pyproject.toml             # depends on: shine
└── README.md
```

### 19. Pipeline Configuration

```yaml
# configs/q1_campaign.yaml

# Tiling
tiling:
  source: mer_catalog
  tile_ids: all                    # or a list of specific IDs
  sub_tile_grid: [4, 4]
  sub_tile_margin: 1.0             # arcminutes

# Data sources
data:
  exposure_archive: s3://nasa-irsa-euclid-q1/q1/VIS/
  catalog_archive: s3://nasa-irsa-euclid-q1/q1/MER/
  scratch_dir: s3://shine-scratch/q1/

# Inference (passed through to SHINE)
inference:
  method: map
  map_config:
    num_steps: 200
    learning_rate: 0.002
  rng_seed: 42

# Galaxy model (passed through to SHINE)
gal:
  type: Exponential
  flux: {type: LogNormal, center: catalog, sigma: 0.5}
  half_light_radius: {type: LogNormal, center: catalog, sigma: 0.3}
  shear:
    type: G1G2
    g1: {type: Normal, mean: 0.0, sigma: 0.05}
    g2: {type: Normal, mean: 0.0, sigma: 0.05}
  ellipticity:
    type: E1E2
    e1: {type: Normal, mean: 0.0, sigma: 0.3}
    e2: {type: Normal, mean: 0.0, sigma: 0.3}
  position:
    type: Offset
    dx: {type: Normal, mean: 0.0, sigma: 0.05}
    dy: {type: Normal, mean: 0.0, sigma: 0.05}

# Output
output:
  catalog_uri: s3://shine-catalog/q1/
  format: parquet
  save_diagnostics: true
  save_model_images: false

# Compute
compute:
  backend: skypilot              # or: modal, slurm, aws_batch
  gpu_type: L4
  use_spot: true
  max_concurrency: 500
```

The `inference` and `gal` sections are passed through to SHINE
unchanged — `shine-pipeline` does not interpret them.

---

## Part IV — Deployment & Cost

### 20. Q1 Case Study (AWS)

#### 20.1 Data Access

The Euclid Q1 data release (~30 TB covering ~63 deg²) is hosted on
Amazon S3 via the
[AWS Registry of Open Data](https://registry.opendata.aws/euclid-q1/)
in the `nasa-irsa-euclid-q1` bucket in `us-east-1`.  Publicly
accessible without authentication (`--no-sign-request`).  Same-region
S3-to-EC2 traffic is free.

```
s3://nasa-irsa-euclid-q1/q1/
  ├── VIS/     # VIS calibrated frames (quadrant FITS)
  ├── NIR/     # NIR frames
  ├── MER/     # Multiwavelength mosaics & catalogs
  ├── SIR/     # Spectroscopic data
  └── RAW/     # Level 1 raw frames
```

#### 20.2 Q1 Footprint

| Field | Area |
|-------|------|
| Euclid Deep Field North | 22.9 deg² |
| Euclid Deep Field South | 28.1 deg² |
| Euclid Deep Field Fornax | 12.1 deg² |
| LDN 1641 | ~1 deg² |
| **Total** | **~65 deg²** |

#### 20.3 Job Count and Compute Time

| Parameter | Value |
|-----------|-------|
| MER tiles (0.25 deg² core) | ~260 |
| Sub-tiles (4 × 4 per tile) | ~4,160 |
| Sources per sub-tile | ~2,000–3,000 |
| Exposures per sub-tile | ~10–30 (deep fields have many more dithers) |

| Inference method | Time per sub-tile | Total GPU-hours |
|-----------------|-------------------|-----------------|
| MAP (~200 steps) | ~20–40 min | ~1,400–2,800 |
| NUTS (500 samples) | ~4–12 hr | ~17,000–50,000 |

#### 20.4 Cost Estimate (MAP)

| Component | SkyPilot Spot | AWS Batch Spot | On-demand |
|-----------|---------------|----------------|-----------|
| GPU compute (~2,100 hrs) | ~$500 | ~$750 | ~$1,700 |
| CPU (Stage 1 + 3) | ~$10 | ~$10 | ~$10 |
| S3 storage | <$5 | <$5 | <$5 |
| **Total** | **~$500** | **~$800** | **~$1,750** |

#### 20.5 Wall-Clock Time (MAP)

| Concurrent GPUs | Wall-clock |
|-----------------|------------|
| 50 | ~42 hours |
| 100 | ~21 hours |
| 200 | ~10 hours |
| 500 | ~4 hours |

#### 20.6 GPU Instance Options

| Instance | GPU | VRAM | On-demand $/hr | Spot $/hr |
|----------|-----|------|----------------|-----------|
| **g6.xlarge** | **NVIDIA L4** | **23 GB** | **$0.80** | **$0.36** |
| g5.xlarge | NVIDIA A10G | 24 GB | $1.01 | $0.40 |
| g6e.xlarge | NVIDIA L40S | 48 GB | $1.86 | ~$0.75 |

Recommended: `g6.xlarge` (L4).  23 GB VRAM is well above the 2–4 GB
requirement.  `g5.xlarge` (A10G) is a viable fallback.

### 21. Full Survey Estimates

| | Q1 | Full Wide Survey |
|---|---|---|
| Area | ~65 deg² | ~15,000 deg² |
| MER tiles | ~260 | ~60,000 |
| Sub-tiles | ~4,200 | ~960,000 |
| MAP GPU-hours | ~2,100 | ~200,000 |
| MAP cost (SkyPilot spot) | ~$500 | ~$50,000 |
| MAP cost (AWS Batch spot) | ~$800 | ~$70,000 |
| NUTS cost (SkyPilot spot) | ~$8,000 | ~$2M |

#### 21.1 Total Cost: Full Wide Survey (MAP)

| Component | Cost |
|-----------|------|
| GPU compute (SkyPilot spot) | ~$50,000 |
| S3 storage (catalog + staging) | ~$500 |
| S3 Intelligent-Tiering (input archive) | ~$5,000/year |
| Infrastructure (Flyte + Grafana, 6 months) | ~$1,000–1,750 |
| **Total** | **~$57,000** |

This is comparable to the cost of a single postdoc-year.

#### 21.2 Throughput

| Resource | Per sub-tile | Full survey (with enough GPUs) |
|----------|-------------|-------------------------------|
| GPU time (MAP, ~200 steps) | ~5–15 min | ~5–15 min |
| GPU time (NUTS, 500 samples) | ~2–6 hours | ~2–6 hours |
| Data I/O | ~1 GB | ~1 PB total |
| Wall-clock (1000 GPUs, MAP) | — | ~2 weeks |

MAP/VI inference is practical at survey scale.  Full NUTS posterior
sampling may be reserved for calibration sub-fields or combined with
amortized initialization strategies.

### 22. Comparison: Academic SLURM vs Modern Stack

| Dimension | SLURM (HPC cluster) | Modern stack |
|-----------|---------------------|-------------|
| Scheduling | `sbatch` + job arrays | Flyte map tasks + SkyPilot |
| GPU provisioning | Fixed cluster allocation | Elastic spot across clouds |
| Spot/preemption | N/A (dedicated nodes) | Automatic cross-cloud recovery |
| Data management | Lustre/GPFS | S3 + Iceberg lakehouse |
| Output catalog | FITS files on disk | Iceberg (versioned, queryable) |
| Monitoring | `squeue`, ad-hoc scripts | FlyteConsole + Grafana |
| Reproducibility | Manual | Git + DVC + ECR + Iceberg snapshots |
| Cost model | Allocation hours (opaque) | Pay-per-second (transparent) |
| Scaling | Limited by cluster | Elastic to thousands of GPUs |
| Setup | Sysadmin dependency | `terraform apply` |

### 23. Phased Adoption

| Phase | Scope | SHINE work | shine-pipeline work |
|-------|-------|------------|---------------------|
| **1. Prototype** | 1 tile (16 sub-tiles) | Multi-CCD data loader, CLI, visibility filtering | Modal scripts, Parquet output, DuckDB analysis |
| **2. Q1 campaign** | 65 deg² (~4,200 sub-tiles) | Validate at scale, tune performance | Flyte workflows, Iceberg catalog, Grafana, Terraform |
| **3. Full survey** | 15,000 deg² (~960K sub-tiles) | Stable — no changes expected | SkyPilot for spot compute, Nessie for catalog releases |

---

## Appendix

### A. Memory Budget (8' Sub-Tile, 4 × 4 Grid)

Assumes 2,500 sources after selection, ~6 overlapping quadrant
exposures, each at full 2048 × 2066 resolution.

| Component | Size | Notes |
|-----------|------|-------|
| Observed images (6 × 2048 × 2066 × 4B) | 96 MB | Full quadrants, unmodified |
| Noise maps (same) | 96 MB | |
| Model images (6 × 2048 × 2066 × 4B) | 96 MB | Forward pass |
| PSF stamps (2,500 × 6 × 21² × 4B) | 26 MB | Dense; only ~4/6 visible per source |
| Source parameters (2,500 × 8 × 4B) | 80 kB | Negligible |
| AD tape (gradient storage) | ~500 MB–1 GB | Depends on method |
| JIT compilation overhead | ~1–2 GB | One-time per config |
| **Total estimate** | **~2–4 GB** | Fits comfortably on L4/A100/H100 |

### B. Glossary

| Term | Definition |
|------|------------|
| MER tile | Euclid sky partition (~32' × 32') used by the MER pipeline |
| Sub-tile | Subdivision of a MER tile (~8' × 8'), the SHINE inference unit |
| Core area | Inner region of a sub-tile where results are reported |
| Extended area | Core + overlap margin, where sources are modeled |
| Quadrant | A VIS CCD quadrant (2048 × 2066 px), loaded as-is |
| Visibility | Whether a source falls within a given quadrant's detector bounds |
| Manifest | YAML file listing quadrant paths + source IDs for one sub-tile |
| ExposureSet | SHINE's stacked array structure for multi-exposure data |

### C. References

**Euclid:**
- [Euclid MER Tile Product (DPDD)](https://euclid.esac.esa.int/msp/dpdd/live/merdpd/dpcards/mer_tile.html)
- [Tiling the Euclid Sky (Kuchner+ 2022, ADASS)](https://ui.adsabs.harvard.edu/abs/2022ASPC..532..329K/abstract)
- [Euclid Q1 Data Release Overview](https://arxiv.org/abs/2503.15302)
- [Euclid Q1 on AWS Open Data](https://registry.opendata.aws/euclid-q1/)

**Workflow orchestration:**
- [Flyte — Kubernetes-native workflow engine](https://flyte.org/)
- [Union.ai — managed Flyte](https://www.union.ai/)
- [Flyte map tasks](https://flyte.org/blog/map-tasks-in-flyte)
- [Flyte spot instance support](https://docs.flyte.org/en/latest/user_guide/productionizing/spot_instances.html)

**GPU compute:**
- [SkyPilot — multi-cloud GPU orchestrator](https://skypilot.readthedocs.io/)
- [SkyPilot managed jobs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html)
- [Modal — serverless GPU compute](https://modal.com/)

**Data lakehouse:**
- [Apache Iceberg — open table format](https://iceberg.apache.org/)
- [Iceberg V3: row lineage](https://opensource.googleblog.com/2025/08/whats-new-in-iceberg-v3.html)
- [DuckDB Iceberg integration](https://duckdb.org/docs/extensions/iceberg.html)
- [Polars — fast DataFrames for Python](https://pola.rs/)
- [S3 Express One Zone](https://aws.amazon.com/s3/storage-classes/express-one-zone/)
- [Project Nessie — Git-like catalog](https://projectnessie.org/)

**Infrastructure:**
- [Terraform](https://www.terraform.io/)
- [Grafana + Loki](https://grafana.com/oss/loki/)
- [DVC — data version control](https://dvc.org/)
- [Kueue — Kubernetes-native job queueing](https://kueue.sigs.k8s.io/)

**JAX / inference:**
- [JAX sharded computation](https://docs.jax.dev/en/latest/sharded-computation.html)
- [NumPyro MCMC](https://num.pyro.ai/en/latest/mcmc.html)
