# SHINE Distributed Inference Specification

## 1. Introduction

### 1.1 Motivation

SHINE currently performs shear inference on a single Euclid VIS quadrant
(~600 sources, 3 exposures) using a single consumer-grade GPU with 10 GB
of VRAM.  Scaling to the full Euclid Wide Survey requires processing
~15,000 deg² of sky containing billions of sources observed across
~50,000 pointings, each with 36 CCDs × 4 quadrants and ~4 dithered
exposures.

This document specifies the distributed inference architecture that
bridges the gap between the single-quadrant prototype and survey-scale
production.

### 1.2 Design Principles

1. **Adopt Euclid's own spatial decomposition.**  The Euclid Science
   Ground Segment (SGS) already partitions the sky into MER tiles
   (32' × 32').  SHINE reuses this tiling rather than inventing its own.

2. **Source-centric, not detector-centric.**  The processing unit is a
   region of the sky (a sub-tile), not a CCD quadrant.  Each sub-tile
   collects all quadrant exposures that overlap its footprint, regardless
   of which CCD or pointing they come from.

3. **Quadrants are immutable data.**  Quadrant images (science, RMS,
   flags) are loaded exactly as delivered by the VIS pipeline — no
   trimming, no reprojection, no pixel modification.  The forward model
   renders galaxies in each quadrant's native pixel coordinate system
   using its own WCS and PSF.  This eliminates an entire class of data
   preparation bugs and keeps the pipeline simple.

4. **Preserve the current pipeline.**  The core rendering code
   (`render_one_galaxy`, tiered `vmap`, scatter-add, Gaussian likelihood)
   and the inference engine (MAP / VI / NUTS) remain unchanged.  New code
   is limited to data preparation and orchestration.

5. **Embarrassingly parallel.**  Sub-tiles are fully independent inference
   problems with no cross-sub-tile communication during inference.

### 1.3 Scope

This specification covers:

- The tile / sub-tile spatial hierarchy
- Data preparation: quadrant discovery and sub-tile assembly
- Boundary handling between sub-tiles
- Visibility-aware source filtering for computational efficiency
- Orchestration on HPC clusters
- Output aggregation into a shear catalog

It does *not* cover changes to the probabilistic model, the rendering
pipeline, or the inference algorithms themselves.

---

## 2. Spatial Hierarchy

### 2.1 MER Tiles (Data Preparation Unit)

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
| Core area encoding | HEALPix indices (Multi-Order Coverage maps) |

The MER tile serves as the **data preparation boundary** for SHINE.
Given a tile ID, the data loader discovers all VIS quadrant exposures
that overlap the tile footprint and loads them.

A MER tile is too large for a single GPU.  At ~475,000 detected sources
per deg² (~30,000–50,000 after SHINE source selection per tile), the
memory requirements for dense per-source-per-exposure arrays exceed
available VRAM.

### 2.2 Sub-Tiles (Inference Unit)

Each MER tile is subdivided into a regular grid of **sub-tiles**.  The
sub-tile is the fundamental inference unit: one sub-tile maps to one GPU
job.

**Baseline configuration: 4 × 4 grid** within each MER tile.

| Property | Value |
|----------|-------|
| Core area | 8' × 8' (4,800 × 4,800 px at 0.1"/px) |
| Extended area | 10' × 10' (6,000 × 6,000 px at 0.1"/px) |
| Overlap margin | 1' per side (600 px) |
| Sources (after selection) | ~2,000–3,000 per sub-tile |
| Overlapping quadrants | ~4–8 (one per dithered pointing per overlapping CCD) |
| Shear model | Constant (g1, g2) per sub-tile |
| GPU memory (estimated) | 2–4 GB |
| Target hardware | NVIDIA L4 (23 GB), A100 (80 GB), or H100 (80 GB) |

The sub-tile grid is configurable.  The sub-tile side length trades off
GPU memory against the number of sources constraining the shear:

| Sub-tile side | Grid | Sources | Shear precision (σ_g/√N) | GPU fit? |
|---------------|------|---------|--------------------------|----------|
| 4' | 8 × 8 | ~500 | ~0.002 | easily |
| 8' | 4 × 4 | ~2,000 | ~0.001 | comfortable |
| 16' | 2 × 2 | ~7,500 | ~0.0006 | tight |
| 32' | 1 × 1 | ~30,000 | ~0.0003 | does not fit |

At 8' per sub-tile the shear field is well-approximated as constant (the
coherence scale of cosmic shear is ~arcminutes), and the per-sub-tile
source count delivers shear precision well below the shape-noise floor.

### 2.3 Hierarchy Diagram

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

---

## 3. Data Preparation Pipeline

### 3.1 Overview

For each MER tile, the data preparation stage:

1. Queries the Euclid archive for all VIS quadrant exposures overlapping
   the tile footprint.
2. Loads each overlapping quadrant as-is (science, RMS, flag images at
   full 2048 × 2066 resolution).
3. Loads the MER source catalog for the tile.
4. Partitions sources into sub-tiles based on sky position.
5. For each sub-tile, assembles an `ExposureSet` from the full quadrant
   images and the sub-tile's source list.

### 3.2 Quadrant Discovery

Given a sub-tile footprint defined by its sky bounding box (RA_min,
RA_max, Dec_min, Dec_max), the loader identifies all quadrant FITS files
whose detector footprint overlaps this bounding box.  The overlap test
uses the quadrant's WCS to project its four corners onto the sky and
check intersection with the sub-tile's extended area.

Each overlapping quadrant is loaded in its entirety — no trimming, no
reprojection, no pixel modification.  All quadrants are full
2048 × 2066 images in their native pixel coordinate system.

Since all VIS quadrants share the same pixel dimensions, they stack
directly into the existing `ExposureSet` layout `(n_exp, 2066, 2048)`
with no padding or reshaping needed.

### 3.3 Source Partitioning

Sources from the MER catalog are assigned to sub-tiles based on their
sky coordinates (RA, Dec):

- A source belongs to a sub-tile's **core** if it falls within the 8' × 8'
  core area.
- A source belongs to a sub-tile's **extended** area if it falls within
  the 10' × 10' region (core + 1' margin on each side).
- Sources in the overlap margin are assigned to the extended area of
  **all** adjacent sub-tiles that contain them.

The extended source list is used for inference (to ensure correct scene
modeling near boundaries).  Results are reported only for core sources.

### 3.4 Per-Sub-Tile ExposureSet Assembly

For each sub-tile, the loader builds an `ExposureSet` containing:

- Full quadrant images (science, RMS, flags) — one per overlapping
  quadrant, unmodified
- Per-source, per-quadrant metadata: pixel positions (projected via each
  quadrant's WCS), WCS Jacobians, interpolated PSF stamps, visibility
  flags
- Catalog-derived quantities: flux, half-light radius, coordinates
- Stamp tier assignments

This is the same data structure and assembly logic used today.  The only
difference is that the set of quadrants comes from multiple CCDs across
multiple pointings, rather than the same CCD across dithered pointings.
Since all VIS quadrants are 2048 × 2066, the stacked array layout is
preserved without modification.

### 3.5 Rendering in Native Quadrant Coordinates

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
5. The stamp is scatter-added onto the model image at the source's pixel
   position.

This is exactly what the existing `_render_tier` and
`_compute_source_metadata` already do.  No coordinate transformation
is needed because each quadrant is processed in its own native frame.

### 3.6 Unmodeled Sources Outside the Sub-Tile

A quadrant may contain sources that fall outside the sub-tile's extended
area.  These sources contribute flux to the observed quadrant image but
are not included in the sub-tile's model.

This does **not** bias the inference.  The likelihood gradient with
respect to any model parameter θ at a pixel p where the model has zero
flux is:

```
∂L/∂θ |_p  =  (obs[p] − model[p]) / σ[p]²  ×  ∂model[p]/∂θ  =  0
```

because `∂model[p]/∂θ = 0` at pixels where no modeled source has a
stamp.  Unmodeled sources create squared residuals in the loss value but
contribute exactly zero gradient, so they affect neither MAP estimates
nor MCMC posterior geometry.

---

## 4. Code Modifications

### 4.1 No Changes to ExposureSet

Because quadrants are loaded as-is at their native 2048 × 2066
resolution, all "exposures" in a sub-tile share the same image
dimensions.  The existing `ExposureSet` dataclass with its stacked
`(n_exp, ny, nx)` arrays works without any modification.

The only new logic is in the data loader, which now discovers and loads
quadrants from multiple CCDs rather than a single CCD across dithered
pointings.

### 4.2 Visibility-Aware Source Filtering

**Current state.**  `_render_tier` vmaps `render_one_galaxy` over all
sources in a stamp-size tier for every exposure, using `source_visible`
to mask invisible sources.  The masked sources still execute the full
FFT convolution pipeline (with dummy parameters), wasting computation.

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
# tier_indices[tier_idx] → current: all sources in this tier
# tier_quad_indices[tier_idx][exp_idx] → new: sources in this tier
#                                        AND visible in this quadrant

tier_quad_indices = []
for t in range(n_tiers):
    tier_sources = np.where(source_stamp_tier == t)[0]
    per_quad = []
    for j in range(n_exp):
        visible = source_visible[tier_sources, j]
        per_quad.append(jnp.array(tier_sources[visible], dtype=jnp.int32))
    tier_quad_indices.append(per_quad)
```

Then in `_render_tier`, simply use the precomputed index array for the
current tier and quadrant:

```python
# In _render_tier: use precomputed indices (static, fixed-size)
indices = tier_quad_indices[tier_idx][exp_idx]

if indices.shape[0] == 0:
    return model_image

# Proceed with vmap over only the visible subset
flux_t = flux[indices]
hlr_t = hlr[indices]
# ...
```

This eliminates wasted FFT computation proportional to the number of
invisible source-exposure pairs.  The index arrays are static constants
with shapes fixed at data preparation time — there is nothing dynamic
during inference, and no impact on JIT compilation.

### 4.3 Exposure Terminology

In the distributed setting, the meaning of "exposure" shifts:

| | Current | Distributed |
|---|---|---|
| What is an exposure? | A full quadrant from one dithered pointing | A full quadrant from one CCD of one pointing |
| How many per inference job? | 3–4 (dithered pointings, same CCD) | ~4–8 (dithered pointings × overlapping CCDs) |
| Image shape | Uniform (2048 × 2066) | Uniform (2048 × 2066) |
| Same CCD across dithers? | Yes (same quadrant ID) | No (different CCDs may cover the same sky region) |
| PSF model per exposure | Single grid (one quadrant) | Different grid per quadrant (from different CCDs) |

The `ExposureSet` data structure and the rendering loop generalize
without change: each "exposure" is simply one full quadrant image (from
one CCD of one pointing) with its associated PSF, WCS, and noise map.
The only difference is how the data loader discovers and selects the
quadrants.

---

## 5. Boundary Handling

### 5.1 The Problem

A source near a sub-tile boundary has a rendering stamp that extends
beyond the sub-tile's core area.  If the neighboring sub-tile does not
model this source, it will see unexplained flux in its observed image
near the boundary, potentially biasing the shear estimate of nearby
core sources whose stamps overlap.

Similarly, blended source groups that straddle a boundary must be
modeled jointly by at least one sub-tile to correctly account for their
overlapping light profiles.

### 5.2 Solution: Core / Extended Areas

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

- **Extended area** (10' × 10'): all sources within this footprint are
  included in the `ExposureSet` and modeled during inference.

- **Core area** (8' × 8'): only sources within this footprint have their
  inferred parameters (shear, flux, size, ellipticity, position)
  written to the output catalog.

- **Overlap margin** (1' per side = 600 px at 0.1"/px): sources in this
  border are modeled by adjacent sub-tiles independently.  Their
  inferred parameters are discarded — they serve only to ensure the
  scene model is complete near the boundary.

Note that the quadrant images themselves extend well beyond the sub-tile
footprint (a full quadrant covers ~3.4' × 3.4').  This is fine — the
model only renders sources in the extended area, and pixels far from any
modeled source contribute zero gradient to the likelihood (Section 3.6).

### 5.3 Margin Sizing

The overlap margin must be large enough that no source in the core area
has its rendering stamp overlapping with an unmodeled source outside the
extended area.  The required margin is:

```
margin ≥ max_stamp_size / 2  (in sky coordinates)
```

With a maximum stamp tier of 256 px:

```
256 / 2 × 0.1"/px = 12.8" ≈ 0.21'
```

A 1' (60") margin provides ~4.7× the minimum required clearance,
accommodating even unusually extended sources and ensuring blend groups
near the boundary are fully contained.

### 5.4 Overlap Overhead

The extended area is (10/8)² = 1.5625× the core area, meaning each
sub-tile models ~56% more sources than strictly needed.  In practice the
overhead is lower because:

- Overlap sources tend to be at the smallest stamp tier (the margin is
  thin relative to the core)
- Rendering cost is dominated by the largest-stamp sources which sit
  deep in the core
- The overlap is shared with 8 neighbors (4 edge + 4 corner), so the
  per-neighbor marginal cost is small

### 5.5 Inter-Tile Boundaries

The same core/extended pattern applies at the MER tile level.  When
SHINE processes sub-tiles near the edge of a MER tile, the sub-tile's
extended area may extend beyond the MER tile boundary into the adjacent
tile's overlap region.  The 2' MER tile overlap (≥ the 1' sub-tile
margin) ensures that source and image data is always available.

---

## 6. Orchestration

### 6.1 Job Structure

Each sub-tile inference is a self-contained job:

```
Input:  tile_id, sub_tile_row, sub_tile_col, config.yaml
Output: shear catalog (core sources only), MAP/posterior parameters,
        diagnostic plots
```

The orchestrator submits jobs to an HPC scheduler (SLURM).  Jobs are
independent and require no inter-job communication.

### 6.2 Pipeline Stages

```
Stage 1: Tile Preparation (CPU, I/O-bound)
──────────────────────────────────────────
  For each MER tile:
  1. Query archive for overlapping VIS quadrant exposures
  2. Load MER source catalog
  3. Partition sources into sub-tiles (core + extended)
  4. For each sub-tile, record which quadrant FITS files to load
  5. Write per-sub-tile manifests to disk
     (list of quadrant paths + source IDs)

Stage 2: Sub-Tile Inference (GPU, compute-bound)
─────────────────────────────────────────────────
  For each sub-tile (1 GPU per sub-tile):
  1. Load manifest (quadrant paths + source list)
  2. Load full quadrant images and PSF grids (unmodified)
  3. Build ExposureSet (project sources onto each quadrant's
     native coordinates, interpolate PSFs, compute visibility)
  4. Construct MultiExposureScene model (unchanged)
  5. Run inference (MAP, VI, or NUTS — unchanged)
  6. Write per-sub-tile results (shear, per-source params, diagnostics)

Stage 3: Catalog Assembly (CPU, I/O-bound)
──────────────────────────────────────────
  1. Collect per-sub-tile shear estimates
  2. Filter to core-area sources only
  3. Assemble into a unified shear catalog
  4. Write as FITS table with tile/sub-tile provenance
```

### 6.3 SLURM Integration

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

python -m shine.distributed.run_subtile \
    --tile-id $TILE_ID \
    --sub-tile-row $SUB_ROW \
    --sub-tile-col $SUB_COL \
    --config configs/euclid_distributed.yaml
```

For processing many tiles, a higher-level script submits one SLURM array
job per tile, or batches multiple tiles into a single large array.

### 6.4 Throughput Estimate

| Resource | Per sub-tile | Per MER tile (16 sub-tiles) | Full survey (~960k sub-tiles) |
|----------|-------------|---------------------------|-------------------------------|
| GPU time (MAP, ~200 steps) | ~5–15 min | ~5–15 min (parallel) | ~5–15 min (with enough GPUs) |
| GPU time (NUTS, 500 samples) | ~2–6 hours | ~2–6 hours (parallel) | ~2–6 hours (with enough GPUs) |
| Data I/O | ~1 GB | ~16 GB | ~1 PB total |
| Wall-clock (1000 GPUs, MAP) | — | — | ~2 weeks |
| Wall-clock (1000 GPUs, NUTS) | — | — | ~6 months |

MAP/VI inference is practical at survey scale.  Full NUTS posterior
sampling may be reserved for calibration sub-fields or combined with
amortized initialization strategies.

---

## 7. Configuration

### 7.1 Distributed Configuration Extension

The existing `EuclidInferenceConfig` is extended with a `distributed`
section:

```yaml
# configs/euclid_distributed.yaml

# ... existing data, sources, gal, inference sections ...

distributed:
  # Tile specification
  tile_source: mer_catalog      # or "custom" for user-defined grid
  tile_id: 102159490            # specific tile, or "all" for batch

  # Sub-tile grid within each MER tile
  sub_tile_grid: [4, 4]         # rows × cols
  sub_tile_margin: 1.0          # overlap margin in arcminutes

  # Data preparation
  exposure_archive: /data/euclid/vis/   # root path for quadrant FITS
  catalog_archive: /data/euclid/mer/    # root path for MER catalogs
  scratch_dir: /scratch/shine/          # intermediate manifests

  # Output
  output_dir: /results/shine/
  output_format: fits            # fits or hdf5
  save_diagnostics: true         # per-sub-tile diagnostic plots
  save_model_images: false       # rendered model images (large)
```

### 7.2 Sub-Tile Metadata

Each sub-tile manifest includes a metadata header:

```yaml
tile_id: 102159490
sub_tile_row: 2
sub_tile_col: 1
core_ra_range: [269.123, 269.256]    # degrees
core_dec_range: [-28.456, -28.323]   # degrees
extended_ra_range: [269.106, 269.273]
extended_dec_range: [-28.473, -28.306]
n_sources_extended: 2847
n_sources_core: 2103
n_quadrants: 7
quadrant_ids: ["3-4.F@exp0", "4-4.E@exp0", "3-4.F@exp1", ...]
```

---

## 8. Output Data Products

### 8.1 Per-Sub-Tile Output

Each inference job produces:

| Product | Format | Contents |
|---------|--------|----------|
| `shear_{tile}_{row}_{col}.fits` | FITS table | g1, g2 estimates with uncertainties for the sub-tile |
| `params_{tile}_{row}_{col}.fits` | FITS table | Per-source MAP/posterior parameters (core sources only) |
| `diagnostics_{tile}_{row}_{col}.png` | PNG | Observed / model / residual panels |
| `metadata_{tile}_{row}_{col}.yaml` | YAML | Sub-tile provenance, runtime, convergence info |

### 8.2 Assembled Shear Catalog

The final data product is a single FITS catalog covering the survey
footprint:

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
offsets) are stored separately per tile to keep the shear catalog
compact.

---

## 9. New Module Structure

```
shine/
├── distributed/                    # New package
│   ├── __init__.py
│   ├── tiling.py                   # MER tile queries, sub-tile grid
│   ├── subtile_loader.py           # Build ExposureSet per sub-tile
│   ├── orchestrator.py             # SLURM job generation and submission
│   ├── catalog_assembler.py        # Merge per-sub-tile outputs
│   └── config.py                   # DistributedConfig pydantic model
├── euclid/
│   ├── config.py                   # Extended with distributed section
│   ├── data_loader.py              # Generalized to load multiple CCDs
│   ├── scene.py                    # Visibility filtering added (§4.2)
│   └── plots.py                    # Unchanged
├── config.py                       # Unchanged
├── inference.py                    # Unchanged
└── prior_utils.py                  # Unchanged
```

### 9.1 Module Responsibilities

**`shine.distributed.tiling`**
- Given a MER tile ID, retrieve its sky footprint (center, core bounds,
  extended bounds) from the Euclid archive or a local tile index.
- Subdivide a tile into a grid of sub-tiles with configurable size and
  overlap margin.
- Assign sources to sub-tiles (core and extended membership).

**`shine.distributed.subtile_loader`**
- Given a sub-tile sky footprint, discover all overlapping quadrant FITS
  files.
- Load full quadrant images and PSF grids (unmodified).
- For each source in the sub-tile's extended area, project its sky
  position onto each quadrant's pixel coordinates, evaluate the local
  WCS Jacobian, interpolate the PSF, and determine visibility.
- Assemble the `ExposureSet`.  This reuses the same per-source metadata
  computation as the existing `EuclidDataLoader._compute_source_metadata`
  but operates across quadrants from different CCDs.

**`shine.distributed.orchestrator`**
- Generate SLURM job scripts or array jobs for batch processing.
- Track job status, handle retries for failed sub-tiles.
- Configurable parallelism (number of concurrent jobs, GPU type).

**`shine.distributed.catalog_assembler`**
- Read per-sub-tile output files.
- Filter to core-area sources.
- Deduplicate any residual overlaps (should not occur with proper core
  assignment, but included as a safety check).
- Write the unified shear catalog.

---

## 10. Implementation Roadmap

### Phase 1: Sub-Tile Data Preparation

- Implement `tiling.py`: MER tile footprint lookup, sub-tile grid
  generation, source partitioning.
- Implement `subtile_loader.py`: discover overlapping quadrants, load
  them as-is, project sources onto each quadrant's native coordinates,
  assemble `ExposureSet`.
- **Validation**: verify that a sub-tile `ExposureSet` built from
  multiple CCD quadrants produces correct MAP results when applied to
  the existing test data region (same quadrant loaded as one of several
  "exposures" should reproduce the single-quadrant result).

### Phase 2: Visibility Filtering

- Add visibility-aware source filtering to `_render_tier` in `scene.py`
  (Section 4.2).
- Benchmark rendering time with and without filtering at sub-tile scale.
- **Validation**: confirm that filtered and unfiltered rendering produce
  identical model images (within floating-point tolerance).

### Phase 3: Orchestration

- Implement `orchestrator.py`: SLURM array job generation.
- Implement `catalog_assembler.py`: merge per-sub-tile outputs.
- End-to-end test on a single MER tile (16 sub-tiles) using the Euclid
  test data.

### Phase 4: Survey-Scale Deployment

- Batch processing across multiple MER tiles.
- Performance profiling and optimization (JIT cache reuse, data I/O
  overlap with computation).
- Shear catalog validation against Euclid Flagship simulations.

---

## 11. Appendix

### A. Memory Budget (8' Sub-Tile, 4 × 4 Grid)

Assumes 2,500 sources after selection, ~6 overlapping quadrant
exposures, each at full 2048 × 2066 resolution.

| Component | Size | Notes |
|-----------|------|-------|
| Observed images (6 × 2048 × 2066 × 4B) | 96 MB | Full quadrants, unmodified |
| Noise maps (same) | 96 MB | |
| Model images (intermediate, 6 × 2048 × 2066 × 4B) | 96 MB | Forward pass |
| PSF stamps (2,500 × 6 × 21² × 4B) | 26 MB | Dense; only ~4/6 visible per source |
| Source parameters (2,500 × 8 × 4B) | 80 kB | Negligible |
| AD tape (gradient storage) | ~500 MB–1 GB | Depends on method |
| JIT compilation overhead | ~1–2 GB | One-time per config |
| **Total estimate** | **~2–4 GB** | Fits comfortably on L4/A100/H100 |

### B. Glossary

| Term | Definition |
|------|------------|
| MER tile | Euclid sky partition (~32' × 32') used by the MER pipeline |
| Sub-tile | Subdivision of a MER tile (~8' × 8') used as the SHINE inference unit |
| Core area | Inner region of a sub-tile where results are reported |
| Extended area | Core + overlap margin, where sources are modeled |
| Quadrant | A VIS CCD quadrant (2048 × 2066 px), loaded as-is without modification |
| Visibility | Whether a source falls within a given quadrant's detector bounds |

### C. References

- [Euclid MER Tile Product (DPDD)](https://euclid.esac.esa.int/msp/dpdd/live/merdpd/dpcards/mer_tile.html)
- [Tiling the Euclid Sky (Kuchner+ 2022, ADASS)](https://ui.adsabs.harvard.edu/abs/2022ASPC..532..329K/abstract)
- [Euclid Q1 Data Release Overview](https://arxiv.org/abs/2503.15302)
- [JAX Parallel Programming & NamedSharding](https://docs.jax.dev/en/latest/sharded-computation.html)
- [shard_map for Manual SPMD Parallelism](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
- [Multi-Controller (Multi-Host) JAX](https://docs.jax.dev/en/latest/multi_process.html)
- [NumPyro MCMC Documentation](https://num.pyro.ai/en/latest/mcmc.html)

---

## 12. AWS Cloud Deployment (Q1 Case Study)

### 12.1 Motivation

The Euclid Q1 data release (~30 TB covering ~63 deg²) is hosted on
Amazon S3 via the [AWS Registry of Open Data](https://registry.opendata.aws/euclid-q1/)
in the `nasa-irsa-euclid-q1` bucket in `us-east-1`.  The bucket is
publicly accessible without authentication (`--no-sign-request`), and
AWS covers both storage and egress costs through its Open Data
Sponsorship Program.

Running SHINE inference in `us-east-1` eliminates all data transfer
costs: S3-to-EC2 traffic within the same region is free.  This makes
AWS a natural deployment target for Q1 processing.

### 12.2 Architecture: AWS Batch + Spot Instances

The SLURM-based orchestration described in Section 6 maps directly to
[AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html),
which provides managed job scheduling with native support for GPU
instances, array jobs, and Spot Instance fleets.

```
┌──────────────────────────────────────────────────────────┐
│  S3: nasa-irsa-euclid-q1  (input, us-east-1, free)      │
│  S3: <user-bucket>        (output + scratch, us-east-1)  │
└──────────┬────────────────────────────────┬───────────────┘
           │                                │
    ┌──────▼───────────┐          ┌─────────▼────────────┐
    │ Stage 1: Tile    │          │ Stage 3: Catalog     │
    │ Preparation      │          │ Assembly             │
    │ AWS Batch (CPU)  │          │ AWS Batch (CPU)      │
    │ c6i.xlarge       │          │ c6i.xlarge           │
    └──────┬───────────┘          └─────────▲────────────┘
           │                                │
    ┌──────▼────────────────────────────────┤
    │ Stage 2: Sub-Tile Inference           │
    │ AWS Batch (GPU, Spot fleet)           │
    │ g6.xlarge (NVIDIA L4, 23 GB VRAM)    │
    │ ~4,200 array jobs, embarrassingly ∥   │
    └───────────────────────────────────────┘
```

**Component mapping from SLURM to AWS Batch:**

| SLURM concept | AWS Batch equivalent |
|---------------|---------------------|
| `sbatch --array=0-15` | Array job with size 16 |
| `--gpus-per-task=1` | Job definition with 1 GPU resource requirement |
| `--partition=gpu` | Compute environment with GPU instance family |
| Retry on node failure | Automatic Spot interruption retry |
| Job dependency (`--dependency`) | Job dependency in `submit_job()` |

### 12.3 Infrastructure Components

**1. Container image (ECR)**

A Docker image containing SHINE, JAX (with CUDA), NumPyro, and
JAX-GalSim, pushed to Amazon Elastic Container Registry.  The image
is built once and reused by all jobs.

**2. Compute environment (Spot fleet)**

A managed compute environment using Spot Instances with the
`SPOT_CAPACITY_OPTIMIZED` allocation strategy, which selects instance
types from the deepest capacity pools to minimize interruption risk.
AWS Batch automatically retries interrupted jobs on fresh instances.

**3. Job definitions**

Three job definitions corresponding to the three pipeline stages:

- **Tile preparation** (CPU): reads quadrant FITS data from S3,
  partitions sources, writes per-sub-tile manifests to S3.
- **Sub-tile inference** (GPU): loads full quadrant images from S3
  (unmodified), builds ExposureSet, runs MAP/VI/NUTS, writes results
  to S3.
- **Catalog assembly** (CPU): merges per-sub-tile outputs into a
  unified shear catalog.

**4. Orchestrator**

A Python script or AWS Step Functions state machine that:

1. Enumerates MER tiles covering the Q1 footprint.
2. Submits Stage 1 array jobs (one element per tile).
3. Submits Stage 2 array jobs with a dependency on Stage 1 (one
   element per sub-tile).
4. Submits Stage 3 once all Stage 2 jobs complete.

The existing `shine.distributed.orchestrator` module (Section 9.1)
would wrap the Batch `submit_job` API instead of generating SLURM
scripts.

**5. Data access**

The data loader reads directly from S3 using `s3fs` or `boto3`
(replacing local file paths).  The `nasa-irsa-euclid-q1` bucket
structure is:

```
s3://nasa-irsa-euclid-q1/q1/
  ├── VIS/     # VIS calibrated frames (quadrant FITS)
  ├── NIR/     # NIR frames
  ├── MER/     # Multiwavelength mosaics & catalogs
  ├── SIR/     # Spectroscopic data
  └── RAW/     # Level 1 raw frames
```

### 12.4 Instance Selection

Each sub-tile requires only 2–4 GB of GPU memory (Section 11,
Appendix A), far below the 80 GB of the A100/H100 targets mentioned
in Section 2.2.  Smaller, cheaper GPU instances are well-suited:

| Instance | GPU | VRAM | vCPU | RAM | On-demand $/hr | Spot $/hr |
|----------|-----|------|------|-----|----------------|-----------|
| **g6.xlarge** | **NVIDIA L4** | **23 GB** | **4** | **16 GB** | **$0.80** | **$0.36** |
| g5.xlarge | NVIDIA A10G | 24 GB | 4 | 16 GB | $1.01 | $0.40 |
| g6e.xlarge | NVIDIA L40S | 48 GB | 4 | 32 GB | $1.86 | ~$0.75 |

**Recommended: `g6.xlarge`** (NVIDIA L4).  The L4 provides strong
FP32 throughput for JAX workloads, 23 GB is well above the 2–4 GB
requirement, and Spot availability in `us-east-1` is good.  The
`g5.xlarge` (A10G) is a viable fallback if L4 Spot capacity is
constrained.

### 12.5 Q1 Scale and Cost Estimate

**Q1 footprint:**

| Field | Area |
|-------|------|
| Euclid Deep Field North | 22.9 deg² |
| Euclid Deep Field South | 28.1 deg² |
| Euclid Deep Field Fornax | 12.1 deg² |
| LDN 1641 | ~1 deg² |
| **Total** | **~65 deg²** |

**Job count:**

| Parameter | Value |
|-----------|-------|
| MER tiles (0.25 deg² core) | ~260 |
| Sub-tiles (4 × 4 per tile) | ~4,160 |
| Sources per sub-tile | ~2,000–3,000 |
| Exposures per sub-tile | ~10–30 (deep fields have many more dithers than the wide survey's ~4) |

**Compute time:**

The deep-field exposure count is higher than the wide survey baseline
of Section 6.4 (~6 quadrants).  Estimated MAP time per sub-tile is
~20–40 minutes.

| Inference method | Time per sub-tile | Total GPU-hours (~4,200 jobs) |
|-----------------|-------------------|-------------------------------|
| MAP (~200 steps) | ~20–40 min | ~1,400–2,800 |
| NUTS (500 samples) | ~4–12 hr | ~17,000–50,000 |

**Cost breakdown (MAP, using g6.xlarge):**

| Component | Spot | On-demand |
|-----------|------|-----------|
| GPU compute (~2,100 GPU-hrs) | $750 | $1,700 |
| CPU (Stage 1 + 3, ~50 hrs) | $10 | $10 |
| EBS scratch storage (~500 GB) | $40 | $40 |
| S3 output storage | <$5 | <$5 |
| S3 data transfer (same region) | $0 | $0 |
| **Total (MAP)** | **~$800** | **~$1,750** |

**Cost breakdown (NUTS, using g6.xlarge):**

| Component | Spot | On-demand |
|-----------|------|-----------|
| GPU compute (~33,000 GPU-hrs) | $12,000 | $26,500 |
| Other (CPU, storage) | ~$100 | ~$100 |
| **Total (NUTS)** | **~$12,000** | **~$27,000** |

MAP inference on Q1 is very affordable.  Full NUTS posterior sampling
is more expensive but may be reserved for calibration sub-fields, with
MAP or VI used for the bulk of the footprint.

### 12.6 Wall-Clock Time (MAP)

| Concurrent GPUs | Wall-clock |
|-----------------|------------|
| 50 | ~42 hours |
| 100 | ~21 hours |
| 200 | ~10 hours |
| 500 | ~4 hours |

AWS Batch Spot fleets can scale to hundreds of instances.  At 200
concurrent `g6.xlarge` Spot instances, Q1 MAP processing completes in
roughly half a day.

### 12.7 Comparison with Full Wide Survey

| | Q1 | Full Wide Survey |
|---|---|---|
| Area | ~65 deg² | ~15,000 deg² |
| MER tiles | ~260 | ~60,000 |
| Sub-tiles | ~4,200 | ~960,000 |
| MAP GPU-hours | ~2,100 | ~240,000 |
| MAP cost (Spot) | ~$800 | ~$86,000 |
| NUTS cost (Spot) | ~$12,000 | ~$2.7M |

Q1 serves as a cost-effective validation run: it exercises the full
distributed pipeline at <1% of the survey-scale cost.
