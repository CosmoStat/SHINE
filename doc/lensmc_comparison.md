# LensMC vs SHINE — Feature Comparison Report

This document compares the official Euclid **LensMC** shear measurement pipeline
(`extern/LensMC/`) with the current **SHINE** implementation (`shine/euclid/`).
The goal is to identify features, calibration strategies, and modelling choices
in LensMC that SHINE does not yet support.

---

## 1. Galaxy Model

| Aspect | LensMC | SHINE |
|--------|--------|-------|
| **Profile** | Bulge + Disc (two-component) | Single Sersic |
| **Bulge** | Sersic with free index `n` (typically 1–6) | — |
| **Disc** | Exponential (Sersic `n = 1`, fixed) | — |
| **Ellipticity** | Shared ellipticity for both components | Per-source `(e1, e2)` |
| **Bulge fraction** | Free parameter `B/T` (bulge-to-total ratio) | — |
| **Flux parameters** | Bulge and disc amplitudes (analytically marginalised) | Single flux per source (sampled) |

**Impact:** The bulge+disc decomposition is a core design choice in LensMC.
Because real galaxies have distinct structural components, a single Sersic
profile may introduce model bias — particularly for bright, well-resolved
galaxies where the bulge/disc ratio affects the ellipticity profile.

---

## 2. Flux Handling and Analytic Marginalisation

### LensMC

LensMC treats the bulge and disc **amplitudes as linear nuisance parameters**
and marginalises over them analytically:

1. Render unit-flux bulge and disc model images (convolved with their
   respective PSFs).
2. Solve for the optimal amplitudes via **non-negative least squares (NNLS)**
   against the observed data.
3. Compute the marginalised log-likelihood directly from the NNLS residual.

This eliminates two parameters per source from the sampling space, dramatically
improving convergence and reducing per-sample cost.

The relevant code is in `lensmc/likelihood.py`:
```python
# Simplified: A = [bulge_model, disc_model], solve A @ x = data
result = scipy.optimize.nnls(A, data_vector)
```

### SHINE

SHINE samples `flux` as a latent variable with a log-normal prior. This means
the sampler must explore flux along with morphology, increasing the effective
dimensionality per source.

**Recommendation:** Implement analytic flux marginalisation. For a single-Sersic
model this reduces to a 1D linear solve per source. For a future bulge+disc
model, NNLS with two components (as in LensMC) would be needed.

---

## 3. Exposure Normalisation

### LensMC

LensMC converts all exposures to a **normalised count rate** before fitting:

```python
# From lensmc/image.py — to_normalised_count_rate()
ncr = (data - background) / (gain * exptime * 10**(zeropoint / -2.5))
```

Each exposure is divided by `gain * exptime * 10^(ZP / -2.5)`, which places
all exposures on a common flux scale (e-/s normalised to a reference
magnitude). A **relative zero-point correction** (`delta_zp`) between
exposures is also applied to account for inter-exposure photometric
variations.

### SHINE

SHINE subtracts the background (from a provided background map or sigma-clipped
median) but does **not** normalise exposures to a common flux scale. The
current approach assumes all exposures share the same gain, exposure time, and
zero point — which is approximately true for dithered Euclid VIS observations
of the same quadrant, but will fail for:

- Exposures with different zero points (e.g., different observing conditions)
- Cross-instrument analyses
- Fields with significant flat-field residuals

**Recommendation:** Add a `to_normalised_count_rate()` step in `EuclidExposure`
or `EuclidDataLoader.load()`. Read gain, exposure time, and zero point from the
FITS headers and normalise before fitting. Store the conversion factor per
exposure in `ExposureSet` for flux interpretation.

---

## 4. PSF Handling

| Aspect | LensMC | SHINE |
|--------|--------|-------|
| **Convolution** | FFTW (pyfftw) with pre-planned transforms | JAX-GalSim FFT convolution |
| **Oversampling** | Renders galaxy at higher resolution, then downsamples with pixel averaging | Native resolution rendering |
| **Separate PSFs** | Different PSFs for bulge and disc components | Single PSF per source per exposure |
| **PSF pixel averaging** | Averages oversampled PSF to native resolution | Not applicable |

### LensMC's oversampled rendering

LensMC renders the galaxy model at `N×` the native pixel resolution (typically
2× or 3×), convolves with a correspondingly oversampled PSF, and then
downsamples via pixel averaging. This accounts for the intra-pixel integration
effect that a native-resolution FFT convolution misses.

```python
# From lensmc/psf.py
psf_oversampled = psf.get_oversampled(factor)
# After convolution at oversampled resolution:
image_native = image_oversampled.reshape(ny, factor, nx, factor).mean(axis=(1, 3))
```

**Recommendation:** JAX-GalSim already handles sub-pixel rendering correctly
through its analytic Fourier-space convolution, so oversampling is less
critical for SHINE. However, verifying that the effective pixel response
function is properly included in the JAX-GalSim rendering would be valuable.

---

## 5. Priors

### LensMC — Informative priors

LensMC uses **informative, physically motivated priors**:

| Parameter | Prior | Reference |
|-----------|-------|-----------|
| Ellipticity `(e1, e2)` | Miller et al. (2013) prior: `p(|e|) ∝ |e| * exp(-|e|^2 / (2σ_e^2)) * (1 - |e|^2)^2` | MNRAS 429, 2858 |
| Size (half-light radius) | Conditional on magnitude: `log(hlr) ~ Normal(μ(mag), σ(mag))` | Calibrated on sims |
| Magnitude | `p(mag) ∝ 10^(α * mag)` (number counts) | Observational counts |
| Bulge fraction `B/T` | `p(B/T) ∝ (1 - B/T)^β` | Calibrated on sims |
| Position offset `(dx, dy)` | Uniform within ±N pixels | Hard bound |
| Sersic index `n` | Uniform within [n_min, n_max] | Hard bound |

### SHINE — Simple Gaussian priors

| Parameter | Prior |
|-----------|-------|
| Shear `(g1, g2)` | `Normal(0, σ_g)` with `σ_g = 0.05` |
| Flux | `LogNormal(log(flux_cat), σ_flux)` |
| Half-light radius | `LogNormal(log(hlr_cat), σ_hlr)` |
| Ellipticity `(e1, e2)` | `Normal(0, σ_e)` with `σ_e = 0.3` |
| Position offset `(dx, dy)` | `Normal(0, σ_pos)` with `σ_pos = 0.05` |

**Key gap:** SHINE's ellipticity prior is a simple Gaussian on each component,
which does not enforce `|e| < 1` and does not match the observed ellipticity
distribution. The Miller et al. (2013) prior is standard in Euclid and
encodes the physical constraint that galaxies cannot have `|e| >= 1`.

**Recommendation:** Implement the Miller et al. (2013) ellipticity prior as a
custom NumPyro distribution. Consider adding a size-magnitude conditional
prior if magnitude information is available in the catalog.

---

## 6. Calibration

### LensMC

LensMC implements two calibration strategies:

#### 6a. Internal Calibration (intcal)

After MCMC sampling, LensMC re-weights the posterior samples using
**importance sampling** to correct for the prior-to-likelihood mismatch
inherent in the original sampling:

```python
# From lensmc/optimise.py — importance sampling for intcal
weight = likelihood(params_new) / likelihood(params_old)  # simplified
```

This corrects the multiplicative and additive shear bias introduced by
noise and model mismatch, without requiring external simulations.

#### 6b. M-calibration (metacalibration-like)

LensMC can compute the **shear response matrix** by:
1. Applying a small artificial shear to the observed image
2. Re-measuring the galaxy on the sheared image
3. Computing `R = d<e>/dg` from the measured ellipticity change

This provides a per-object calibration factor that corrects multiplicative
bias at the catalog level.

### SHINE

SHINE has **no calibration infrastructure**. The MAP or NUTS estimates are
reported as-is, without bias correction.

**Recommendation:** This is a significant gap for science-grade shear
measurement. At minimum, implement a framework for:
1. Running SHINE on image simulations with known shear to measure bias
   (the standard `m` and `c` calibration)
2. Consider implementing analytic response estimation (metacalibration-style)
   using JAX's autodiff capabilities — this is a natural fit for a
   differentiable pipeline

---

## 7. Segmentation and Neighbour Handling

### LensMC

LensMC uses **SExtractor segmentation maps** to:
- Identify which pixels belong to the target object vs neighbours
- Mask neighbour pixels during fitting
- Optionally fit blended objects **jointly** with coupled position constraints

```python
# From lensmc/image.py
seg_id = segmentation_map[stamp_slice]
mask = (seg_id != 0) & (seg_id != target_id)  # mask neighbours
```

For severe blends, LensMC can fit multiple objects simultaneously, sharing
a common image model with separate morphological parameters per object.

### SHINE

SHINE renders all sources into a shared image via `scatter_add` but does
**not** use segmentation maps. All sources are modelled simultaneously in
the forward model (which is conceptually correct for a scene-level approach),
but there is no mechanism to:
- Identify and mask contaminating objects not in the source catalog
- Weight down pixels dominated by bright neighbours
- Handle objects at stamp edges

**Note:** SHINE's scene-modelling approach is fundamentally different from
LensMC's object-by-object fitting. By modelling all sources in the field
simultaneously, SHINE implicitly handles blends — provided all relevant
sources are included in the catalog. The gap is for **uncatalogued** objects
(stars, artefacts, faint sources below the detection threshold).

**Recommendation:** Consider adding a residual masking step: after a first
MAP pass, identify pixels with large residuals that could indicate
uncatalogued sources, and down-weight or mask them in subsequent inference.

---

## 8. Background Estimation

| Aspect | LensMC | SHINE |
|--------|--------|-------|
| **Method** | DC level + linear gradient fit per stamp | Global background map or sigma-clipped median |
| **Gradient** | Fits `bg(x,y) = a + b*x + c*y` per stamp | Not supported |
| **Per-stamp** | Yes — local background per object | No — global subtraction |

LensMC estimates a local background for each postage stamp by fitting a
plane to the unmasked border pixels. This accounts for spatial variations
in the background that a global map may miss.

**Recommendation:** For SHINE's scene-level approach, the global background
subtraction is reasonable. However, adding a per-exposure constant background
offset as a free parameter in the model could improve robustness.

---

## 9. Goodness of Fit and Quality Control

### LensMC

- **F-test:** Compares the chi-squared of the best-fit model against the
  chi-squared of a background-only model. Objects with F-test p-value below
  a threshold are flagged as poor fits.
- **Contamination detection:** Compares model images across exposures to
  detect exposures where an unmodelled source contaminates the stamp.
  Contaminated exposures are excluded from the joint fit.
- **Convergence diagnostics:** Monitors chain convergence via acceptance rate
  and inter-chain variance.

### SHINE

- Reports chi-squared per pixel in the notebook visualisation.
- No automated quality flagging or contamination detection.

**Recommendation:** Add per-source goodness-of-fit metrics (reduced chi-squared,
model evidence) to the output. Flag sources with poor fits for downstream
exclusion.

---

## 10. Sampling Strategy

| Aspect | LensMC | SHINE |
|--------|--------|-------|
| **Sampler** | Custom Metropolis-Hastings | NumPyro NUTS (HMC) |
| **Annealing** | Simulated annealing for burn-in | NUTS warmup (automatic step-size/mass-matrix) |
| **Parallel tempering** | Multiple temperature chains | Not supported |
| **Affine-invariant** | Optional emcee-style ensemble moves | Not supported |
| **MAP** | Not used (direct MCMC) | SVI with AutoDelta guide |

LensMC uses a custom MH sampler with several enhancements:
- **Simulated annealing** during burn-in (temperature schedule)
- **Parallel annealing** with multiple temperature chains that swap states
- **Affine-invariant** ensemble moves (stretch moves à la emcee)

SHINE uses NumPyro's NUTS sampler, which is generally superior for
continuous parameter spaces due to gradient-based proposals. The main
advantage of NUTS is that it avoids random-walk behaviour and scales
better with dimensionality.

**Assessment:** SHINE's choice of NUTS is arguably better than LensMC's MH
for this problem class. No action needed.

---

## 11. Astrometric Distortion

### LensMC

LensMC applies the full **2×2 astrometric distortion matrix** per exposure
when rendering galaxy models. This matrix encodes:
- Pixel scale variations across the detector
- Rotation between sky and pixel coordinates
- Shear distortion from the optical system

```python
# From lensmc/image.py
# distortion_matrix = [[dudx, dudy], [dvdx, dvdy]]
# Applied to galaxy profile before rendering
```

### SHINE

SHINE computes and stores the WCS Jacobian per source per exposure
(`wcs_jacobians` in `ExposureSet`) and applies it in the scene model.
This is functionally equivalent to LensMC's approach.

**Assessment:** Feature parity. No action needed.

---

## 12. Star-Galaxy Separation

### LensMC

LensMC applies a size-based threshold to exclude point sources:
```python
# Objects with half-light radius < threshold * PSF_FWHM are classified as stars
if hlr < star_threshold * psf_fwhm:
    skip_object()
```

### SHINE

SHINE filters based on `VIS_DET_TYPE != 2` (point source flag) from the
MER catalog, plus optional size cuts.

**Assessment:** Functionally equivalent — both exclude unresolved objects.

---

## Summary of Gaps (Priority-Ordered)

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| **High** | Exposure normalisation (gain, exptime, ZP) | Low | Required for correctness across exposures |
| **High** | Analytic flux marginalisation | Medium | Major efficiency gain, reduces dimensionality |
| **High** | Miller et al. (2013) ellipticity prior | Low | Standard in Euclid, prevents `\|e\| >= 1` |
| **High** | Shear calibration framework (`m`, `c` bias) | Medium | Required for science-grade results |
| **Medium** | Bulge+disc galaxy model | High | Reduces model bias for resolved galaxies |
| **Medium** | Per-source goodness-of-fit metrics | Low | Quality control for downstream analysis |
| **Medium** | Size-magnitude conditional prior | Low | Regularises size estimates |
| **Low** | Residual-based contamination masking | Medium | Handles uncatalogued sources |
| **Low** | Background gradient per exposure | Low | Minor improvement for scene-level approach |
| **Low** | Metacalibration / response estimation | High | Powerful but complex; JAX autodiff is a natural fit |

---

## Conclusion

SHINE's scene-level forward-modelling approach is architecturally more
principled than LensMC's object-by-object fitting — by modelling all sources
simultaneously, it naturally handles blends and shared parameters (like shear).
The use of JAX and NumPyro provides automatic differentiation and modern
inference algorithms (NUTS) that LensMC lacks.

The most impactful gaps to close are:

1. **Exposure normalisation** — straightforward to implement, required for
   multi-exposure correctness
2. **Analytic flux marginalisation** — significant efficiency gain with
   moderate implementation effort
3. **Ellipticity prior** — simple to add, standard in Euclid pipelines
4. **Calibration framework** — essential for science applications, and
   SHINE's differentiable design makes metacalibration-style approaches
   particularly natural
