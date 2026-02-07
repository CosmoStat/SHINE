# Euclid Flagship 2 Galaxy Mock Catalogue — Technical Reference

## Overview

The **Euclid Flagship 2** (FS2) galaxy mock catalogue is the largest synthetic galaxy catalogue ever built, designed to support the scientific exploitation of ESA's Euclid mission. It is described in detail in:

> **Euclid V.** Castander, F. J., Fosalba, P., et al. (2025). *The Flagship galaxy mock catalogue: A comprehensive simulation for the Euclid mission.* A&A, 697, A5.  
> [arXiv:2405.13495](https://arxiv.org/abs/2405.13495) — [DOI:10.1051/0004-6361/202450853](https://doi.org/10.1051/0004-6361/202450853)

The catalogue is **publicly available** (Open Access, CC BY 4.0) on the **CosmoHub** platform since September 2025.

---

## Simulation Specifications

| Parameter | Value |
|---|---|
| N-body simulation | 4 trillion (4×10¹²) dark matter particles |
| Box size | 3600 h⁻¹ Mpc on a side |
| Particle mass | mₚ = 10⁹ h⁻¹ M☉ |
| Halo finder | ROCKSTAR (Behroozi et al. 2013) |
| Lightcone | Generated on the fly, one octant of the sky |
| Max redshift | z = 3 |
| Haloes | 16 billion main haloes (all-sky: 126 billion) |
| Galaxies | **3.4 billion** (magnitude cut H_E < 26) |
| Properties per galaxy | **~400** modelled properties |
| Cosmology | Planck 2015 ΛCDM (Ωm=0.319, ΩΛ=0.681, h=0.67, σ₈=0.83, ns=0.96) |
| Galaxy population method | HOD + Abundance Matching, calibrated against SDSS, COSMOS |
| Version | 2.1.10 (current public release) |

---

## Access: CosmoHub Platform

### URLs

- **Galaxy mock catalogue**: <https://cosmohub.pic.es/catalogs/353>
- **Dark matter halo catalogue**: <https://cosmohub.pic.es/catalogs/352>

### Registration & Access

CosmoHub is operated by the Port d'Informació Científica (PIC) in Barcelona. Access requires:

1. **Free registration** at <https://cosmohub.pic.es> — open to the scientific community
2. Accounts are approved (not instant), typically within a few days

### How to Query & Download

CosmoHub provides:

- **SQL-like query interface**: Write custom queries selecting specific columns and applying filters (e.g., spatial, magnitude, redshift cuts). This is the primary interaction mode.
- **Online plotting**: Preview distributions before downloading.
- **Bulk download**: Export results as CSV or compressed files. Downloads are queued and processed asynchronously for large queries.
- **HEALPix spatial indexing**: The catalogue is indexed by HEALPix pixel (NSIDE=512, nested scheme), enabling efficient spatial queries on sub-regions of sky.

### Practical Data Volume

The full catalogue is enormous (~15 TB total). For scene simulation work, you would typically query a small sky patch. For example:

```sql
SELECT ra_gal, dec_gal, z_obs, bulge_fraction, bulge_r50, disk_r50,
       disk_angle, inclination_angle, sed_template,
       euclid_vis, euclid_y, euclid_j, euclid_h,
       kappa, gamma1, gamma2
FROM flagship2_galaxies
WHERE hpix_512 BETWEEN 100000 AND 100010
  AND euclid_vis < 25.0
```

This would return a manageable subset for a few square degrees.

---

## Catalogue Format

The catalogue is stored in **FITS format** internally on CosmoHub's Hadoop platform, and exposed via SQL queries. Downloads are typically in **CSV** (or compressed CSV). Each row is one galaxy.

---

## Galaxy Properties — Complete Column Groups

### 1. Identification & Position

| Column | Description |
|---|---|
| `source_id` | Unique galaxy identifier |
| `ra_gal` | Right Ascension (degrees) |
| `dec_gal` | Declination (degrees) |
| `x, y, z` | Comoving Cartesian coordinates (Mpc/h) |
| `vx, vy, vz` | Peculiar velocities (km/s) |
| `hpix_512` | HEALPix pixel index (NSIDE=512, nested) |

### 2. Redshift

| Column | Description |
|---|---|
| `z_cgal` | True (cosmological) redshift |
| `z_obs` | Observed redshift (including peculiar velocity) |

### 3. Photometry — Magnitudes (AB system)

Apparent magnitudes (AB) in multiple bands, **including lensing magnification**. Intrinsic fluxes can be recovered by dividing by the magnification factor μ = (1−κ)² − γ₁² − γ₂².

**Euclid bands:**

| Column | Band |
|---|---|
| `euclid_vis` | Euclid VIS (broad optical, ~550–900 nm) |
| `euclid_y` | Euclid NISP Y |
| `euclid_j` | Euclid NISP J |
| `euclid_h` | Euclid NISP H (H_E) |

**Ground-based bands** (for complementary surveys like LSST/Rubin):

| Column | Band |
|---|---|
| `des_asahi_full_g` | g-band |
| `des_asahi_full_r` | r-band |
| `des_asahi_full_i` | i-band |
| `des_asahi_full_z` | z-band |
| `lsst_u` | LSST u-band |

Additional bands include SDSS, CFHT, Subaru HSC, VISTA, and others — over 30 filter bandpasses in total.

**Separate bulge and disk magnitudes** are also available per band (e.g., `euclid_vis_bulge`, `euclid_vis_disk`), enabling bulge+disk decomposed photometry for scene rendering.

### 4. Spectral Energy Distributions (SEDs)

| Column | Description |
|---|---|
| `sed_template` | Index into the SED template library |
| `sed_template_bulge` | SED template index for the bulge component |
| `sed_template_disk` | SED template index for the disk component |
| `extinction_bulge` | Internal extinction A_V for the bulge |
| `extinction_disk` | Internal extinction A_V for the disk |

The SED templates are a library of ~100 rest-frame spectral templates derived from stellar population synthesis models (Bruzual & Charlot 2003 type). Each galaxy's observed SED is reconstructed as:

```
observed_SED(λ) = SED_template(λ/(1+z)) × extinction_law(A_V, R_V) × (1+z) factor
```

with separate bulge and disk templates combined weighted by the bulge fraction. The SED library itself is distributed with the Euclid simulation pipeline; it is referenced by index in the catalogue.

### 5. Emission Lines

Emission line fluxes (log₁₀ in erg/cm²/s), calibrated using the Pozzetti model:

| Column | Line |
|---|---|
| `halpha_logflam_ext_mag` | Hα (6563 Å) — **observed, includes extinction** |
| `hbeta_logflam_ext_mag` | Hβ (4861 Å) |
| `o2_logflam_ext_mag` | [OII] (3727 Å) |
| `o3_logflam_ext_mag` | [OIII] (5007 Å) |
| `n2_logflam_ext_mag` | [NII] (6584 Å) |
| `s2_logflam_ext_mag` | [SII] (6717/6731 Å) |

Intrinsic (unextincted) versions of these are also available.

### 6. Morphology & Shape

This is the key section for scene simulation. The galaxy model is a **bulge + disk decomposition**:

- **Bulge**: de Vaucouleurs profile (Sérsic n=4)
- **Disk**: Exponential profile (Sérsic n=1), with 3D inclination

| Column | Description |
|---|---|
| `bulge_fraction` | B/T — ratio of bulge flux to total flux |
| `bulge_r50` | Bulge half-light radius, major axis (arcsec) |
| `bulge_ellipticity` | Bulge ellipticity (1 − b/a) |
| `disk_r50` | Disk half-light radius, major axis (arcsec) |
| `disk_ellipticity` | Disk ellipticity |
| `disk_scalelength` | Exponential scale length of the disk (arcsec) |
| `inclination_angle` | Galaxy inclination (0° = face-on, 90° = edge-on) |
| `disk_angle` | Position angle of disk rotation axis (degrees, N→E) |

**Important notes for rendering:**

- The bulge position angle is assumed equal to the disk position angle (`bulge_angle = disk_angle`).
- Sérsic index is **fixed**: n=4 for bulges, n=1 for disks. There is no free Sérsic index column.
- These profiles are directly compatible with **GalSim** rendering: use `galsim.DeVaucouleurs` for bulge and `galsim.InclinedExponential` for disk, or the standard `galsim.Sersic(n=4)` + `galsim.Sersic(n=1)`.
- Ellipticities for bulge and disk are computed following the recipe in the Euclid SHE-SIM documentation.
- QSO/AGN sources are treated as point sources (morphology columns ignored).

### 7. Weak Lensing Properties

| Column | Description |
|---|---|
| `kappa` | Convergence κ |
| `gamma1` | Shear component γ₁ |
| `gamma2` | Shear component γ₂ |
| `magnification` | Lensing magnification factor μ |

**Convention warning**: Flagship assumes a sign flip in the declination axis (+DEC → −DEC), meaning there is a sign flip in γ₁ and γ₂ compared to the usual weak lensing convention. This is documented in Euclid Redmine issue #10560.

### 8. Stellar & Physical Properties

| Column | Description |
|---|---|
| `stellar_mass` | Total stellar mass (M☉) |
| `sfr` | Star formation rate (M☉/yr) |
| `metallicity` | Stellar metallicity |
| `age` | Mean stellar age |

### 9. Halo Properties

| Column | Description |
|---|---|
| `halo_id` | Host halo identifier |
| `halo_mass` | Host halo virial mass M_vir (M☉/h) |
| `is_central` | Central (1) vs satellite (0) galaxy flag |
| `halo_ra, halo_dec` | Halo position on sky |
| `halo_z` | Halo redshift |

---

## How the Catalogue Was Built — Pipeline Summary

1. **N-body simulation**: 4 trillion DM particles run on Piz Daint supercomputer (Swiss CSCS). Lightcone output on the fly.

2. **Halo finding**: ROCKSTAR halo finder on the lightcone particle data → 16 billion haloes in one octant.

3. **Galaxy population via HOD + Abundance Matching**:
   - Central galaxies assigned to haloes using a Halo Occupation Distribution.
   - Satellite galaxies distributed following NFW profiles within haloes.
   - Luminosities assigned via Subhalo Abundance Matching (SHAM), calibrated against the observed luminosity function.

4. **SED assignment**: Galaxy colours and SEDs assigned by matching to observed colour-magnitude-redshift relations from SDSS and COSMOS. Separate bulge and disk SEDs assigned based on morphological type.

5. **Morphology**: Bulge fraction, sizes, ellipticities, and inclinations assigned from empirical scaling relations calibrated against observations.

6. **Emission lines**: Assigned using the Pozzetti et al. model, calibrated to observed Hα luminosity functions.

7. **Lensing**: Convergence and shear computed from the DM particle distribution using HEALPix mass maps (ray-tracing through the lightcone).

8. **Validation**: Extensive comparison against SDSS, COSMOS, GAMA, and other observational datasets for number counts, luminosity functions, colour distributions, angular correlation functions, and cosmic shear statistics.

---

## Suitability for Scene Simulation

### What you get ✅

- Realistic galaxy positions with correct clustering (HOD-based, validated against observations)
- Bulge+disk parametric decomposition directly usable with GalSim or similar rendering tools
- Multi-band photometry (Euclid VIS/NISP + ground bands) including lensing magnification
- SED templates for chromatic rendering
- Emission line fluxes for spectroscopic simulations
- Weak lensing shear/convergence per galaxy for shape distortions
- Complete down to H_E < 26 (well beyond Euclid Wide limit of ~24.5)

### What you don't get ⚠️

- **No free Sérsic index**: Bulges are always n=4, disks always n=1. No irregular, clumpy, or merging morphologies.
- **No real galaxy stamps**: This is a parametric catalogue, not a library of observed galaxy images. For realistic pixel-level morphologies with substructure, complement with the **GalSim COSMOS training sample** (real HST galaxy cutouts).
- **No stars**: The star catalogue is a separate product.
- **No PSF model**: The catalogue provides input truth; PSF convolution is handled by the image simulation pipeline (ELViS/SIM).
- **SED library not directly in CosmoHub**: The SED templates are referenced by index. You need the actual SED library files (distributed with the Euclid simulation software, or reconstructable from Bruzual & Charlot models).

### Complementary Resources

| Resource | Use |
|---|---|
| **GalSim COSMOS** | Real galaxy stamps for realistic morphologies |
| **Euclid Q1 data** | Real Euclid observations (available via ESA Science Archive) |
| **CosmoDC2** (DESC) | Comparable LSST-oriented mock catalogue |
| **MICE simulation** | Earlier mock from same team, also on CosmoHub |

---

## Citation

When using this catalogue, cite:

```bibtex
@article{EuclidCollaboration2025Flagship,
  author  = {{Euclid Collaboration}: Castander, F. J. and Fosalba, P. and Stadel, J. and Potter, D. and Carretero, J. and others},
  title   = {Euclid. V. The Flagship galaxy mock catalogue: A comprehensive simulation for the Euclid mission},
  journal = {Astronomy \& Astrophysics},
  volume  = {697},
  pages   = {A5},
  year    = {2025},
  doi     = {10.1051/0004-6361/202450853}
}
```

And acknowledge CosmoHub:

> *CosmoHub has been developed by the Port d'Informació Científica (PIC), maintained through a collaboration of IFAE, CIEMAT, and ICE-CSIC (Carretero et al. 2017; Tallada et al. 2020).*