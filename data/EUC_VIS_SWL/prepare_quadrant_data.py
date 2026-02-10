"""
Download and extract a single VIS quadrant (3-4.F) from the Euclid Q1 deep
exposures around NGC 6505, plus matching background maps, PSF stamp, and a
merged source catalogue covering the quadrant footprint.

Outputs (in OUTPUT_DIR):
  - 3 single-quadrant FITS files (PRIMARY + SCI/RMS/FLG), one per dither
  - 3 background map FITS files (PRIMARY + BKG), one per dither
  - 1 PSF FITS file with the 3-4.F stamp
  - 1 source catalogue (FITS table) from the MER merged catalogue

Skips downloads/extractions if the output files already exist on disk.
"""

from astroquery.esa.euclid import Euclid
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import astropy.units as u
import numpy as np
import os
import glob

# ── Configuration ──────────────────────────────────────────────
TARGET     = 'NGC 6505'
RADIUS     = u.Quantity(0.5, u.deg)
QUADRANT   = '3-4.F'                     # quadrant closest to pointing center
DATA_DIR   = '/content'                   # where full-frame files live / get downloaded
OUTPUT_DIR = '/content/quadrant_data'     # where trimmed files go
# ───────────────────────────────────────────────────────────────

# The three deep (560s) dither exposures that mutually overlap
DITHER_KEYS = ['002704-00-1', '002704-01-1', '002704-02-1']

os.makedirs(OUTPUT_DIR, exist_ok=True)
Euclid.ROW_LIMIT = -1

# ── Step 1: Find or download the full-frame files ─────────────
print("[1/7] Locating full-frame VIS files ...")

# Check what's already on disk
existing = glob.glob(os.path.join(DATA_DIR, 'EUC_VIS_SWL-DET-*.fits'))
existing_keys = {os.path.basename(f): f for f in existing}

# Match files to our dither keys
needed_downloads = []
frame_files = {}
for key in DITHER_KEYS:
    matches = [f for name, f in existing_keys.items() if key in name]
    if matches:
        frame_files[key] = matches[0]
        print(f"  FOUND on disk: {os.path.basename(matches[0])}")
    else:
        needed_downloads.append(key)
        print(f"  MISSING: dither key {key}, will query archive")

# Download any missing files
if needed_downloads:
    print(f"\n  Querying archive for {len(needed_downloads)} missing file(s) ...")
    job = Euclid.cone_search(
        coordinate=TARGET, radius=RADIUS,
        table_name="q1.calibrated_frame",
        ra_column_name="ra", dec_column_name="dec",
        async_job=True,
        columns=['file_name', 'instrument_name', 'observation_id']
    )
    res = job.get_results()
    vis = res[res['instrument_name'] == 'VIS']

    for key in needed_downloads:
        matches = [r for r in vis if key in r['file_name']]
        if not matches:
            print(f"  WARNING: No archive match for key {key}, skipping")
            continue
        fname = matches[0]['file_name']
        outpath = os.path.join(DATA_DIR, fname)
        print(f"  Downloading: {fname}")
        Euclid.get_product(file_name=fname, output_file=outpath)
        frame_files[key] = outpath

# ── Step 2: Find or download background map files ─────────────
print("\n[2/7] Locating background map files ...")

bkg_existing = glob.glob(os.path.join(DATA_DIR, 'EUC_VIS_SWL-BKG-*.fits'))
bkg_existing_keys = {os.path.basename(f): f for f in bkg_existing}

bkg_needed = []
bkg_files = {}
for key in DITHER_KEYS:
    matches = [f for name, f in bkg_existing_keys.items() if key in name]
    if matches:
        bkg_files[key] = matches[0]
        print(f"  FOUND on disk: {os.path.basename(matches[0])}")
    else:
        bkg_needed.append(key)
        print(f"  MISSING: BKG for dither key {key}")

if bkg_needed:
    print("  Querying archive for BKG files ...")
    job_bkg = Euclid.launch_job_async("""
        SELECT DISTINCT file_name, observation_id
        FROM q1.aux_calibrated
        WHERE instrument_name = 'VIS'
          AND stype = 'BKG'
          AND observation_id = '2704'
    """)
    bkg_res = job_bkg.get_results()

    for key in bkg_needed:
        matches = [r for r in bkg_res if key in r['file_name']]
        if not matches:
            print(f"  WARNING: No BKG match for key {key}, skipping")
            continue
        fname = matches[0]['file_name']
        outpath = os.path.join(DATA_DIR, fname)
        print(f"  Downloading: {fname}")
        Euclid.get_product(file_name=fname, output_file=outpath)
        bkg_files[key] = outpath

# ── Step 3: Find or download the PSF file ─────────────────────
print("\n[3/7] Locating PSF model file ...")

psf_existing = glob.glob(os.path.join(DATA_DIR, 'psf_models', 'EUC_VIS_GRD-PSF-*.fits'))
if not psf_existing:
    psf_existing = glob.glob(os.path.join(DATA_DIR, 'EUC_VIS_GRD-PSF-*.fits'))

if psf_existing:
    psf_full_path = psf_existing[0]
    print(f"  FOUND on disk: {os.path.basename(psf_full_path)}")
else:
    print("  Querying archive for PSF file ...")
    job_psf = Euclid.launch_job_async("""
        SELECT DISTINCT file_name
        FROM q1.aux_calibrated
        WHERE instrument_name = 'VIS'
          AND stype = 'PSF MODEL'
          AND observation_id = '2704'
    """)
    psf_res = job_psf.get_results()
    psf_fname = psf_res['file_name'][0]
    psf_full_path = os.path.join(DATA_DIR, psf_fname)
    print(f"  Downloading: {psf_fname}")
    Euclid.get_product(file_name=psf_fname, output_file=psf_full_path)

# ── Step 4: Extract quadrant from each exposure ───────────────
print(f"\n[4/7] Extracting quadrant {QUADRANT} from each exposure ...")

sci_extensions = [f'{QUADRANT}.SCI', f'{QUADRANT}.RMS', f'{QUADRANT}.FLG']
quadrant_files = []

for key in DITHER_KEYS:
    if key not in frame_files:
        continue

    src = frame_files[key]
    basename = os.path.basename(src).replace('.fits', f'_{QUADRANT.replace(".", "-")}.fits')
    dst = os.path.join(OUTPUT_DIR, basename)
    quadrant_files.append(dst)

    if os.path.exists(dst):
        print(f"  SKIP (exists): {basename}")
        continue

    print(f"  Extracting from {os.path.basename(src)} -> {basename}")
    with fits.open(src) as hdul:
        new_hdus = [fits.PrimaryHDU(header=hdul[0].header)]
        for ext in sci_extensions:
            new_hdus.append(hdul[ext].copy())
        new_hdul = fits.HDUList(new_hdus)
        new_hdul.writeto(dst, overwrite=True)

# ── Step 5: Extract background quadrant from each BKG file ────
print(f"\n[5/7] Extracting background map for quadrant {QUADRANT} ...")

for key in DITHER_KEYS:
    if key not in bkg_files:
        continue

    src = bkg_files[key]
    basename = os.path.basename(src).replace('.fits', f'_{QUADRANT.replace(".", "-")}.fits')
    dst = os.path.join(OUTPUT_DIR, basename)

    if os.path.exists(dst):
        print(f"  SKIP (exists): {basename}")
        continue

    print(f"  Extracting from {os.path.basename(src)} -> {basename}")
    with fits.open(src) as hdul:
        new_hdus = [fits.PrimaryHDU(header=hdul[0].header)]
        new_hdus.append(hdul[QUADRANT].copy())
        new_hdul = fits.HDUList(new_hdus)
        new_hdul.writeto(dst, overwrite=True)

# ── Step 6: Extract PSF stamp for this quadrant ──────────────
print(f"\n[6/7] Extracting PSF stamp for quadrant {QUADRANT} ...")

psf_dst = os.path.join(OUTPUT_DIR, f'PSF_{QUADRANT.replace(".", "-")}.fits')
if os.path.exists(psf_dst):
    print(f"  SKIP (exists): {os.path.basename(psf_dst)}")
else:
    with fits.open(psf_full_path) as hdul:
        new_hdus = [fits.PrimaryHDU(header=hdul[0].header)]
        new_hdus.append(hdul[QUADRANT].copy())
        new_hdul = fits.HDUList(new_hdus)
        new_hdul.writeto(psf_dst, overwrite=True)
    print(f"  Wrote: {os.path.basename(psf_dst)}")

# ── Step 7: Download merged source catalogue ─────────────────
cat_dst = os.path.join(OUTPUT_DIR, f'catalogue_{QUADRANT.replace(".", "-")}.fits')

if os.path.exists(cat_dst):
    print(f"\n[7/7] Source catalogue already exists: {os.path.basename(cat_dst)}")
    cat = Table.read(cat_dst)
else:
    print(f"\n[7/7] Querying MER merged catalogue for quadrant footprint ...")

    # Compute the union bounding box of all extracted quadrants
    ra_all, dec_all = [], []
    for qf in quadrant_files:
        with fits.open(qf) as hdul:
            h = hdul[f'{QUADRANT}.SCI'].header
            w = WCS(h)
            nx, ny = h['NAXIS1'], h['NAXIS2']
            corners = np.array([[0,0],[nx,0],[nx,ny],[0,ny]], dtype=float)
            ra, dec = w.all_pix2world(corners[:,0], corners[:,1], 0)
            ra_all.extend(ra)
            dec_all.extend(dec)

    ra_min, ra_max = min(ra_all), max(ra_all)
    dec_min, dec_max = min(dec_all), max(dec_all)
    print(f"  Footprint bounding box: RA=[{ra_min:.4f}, {ra_max:.4f}], DEC=[{dec_min:.4f}, {dec_max:.4f}]")

    query = f"""
    SELECT object_id, right_ascension, declination,
           right_ascension_psf_fitting, declination_psf_fitting,
           flux_vis_2fwhm_aper, fluxerr_vis_2fwhm_aper,
           flux_vis_3fwhm_aper, fluxerr_vis_3fwhm_aper,
           flux_vis_psf, fluxerr_vis_psf,
           flux_detection_total, fluxerr_detection_total,
           flux_segmentation, fluxerr_segmentation,
           semimajor_axis, semimajor_axis_err,
           ellipticity, ellipticity_err,
           position_angle, position_angle_err,
           kron_radius, kron_radius_err, fwhm,
           point_like_flag, point_like_prob,
           extended_flag, extended_prob,
           spurious_flag, spurious_prob,
           flag_vis, vis_det,
           deblended_flag, det_quality_flag,
           mu_max, mumax_minus_mag, segmentation_area
    FROM catalogue.mer_catalogue
    WHERE right_ascension BETWEEN {ra_min} AND {ra_max}
      AND declination BETWEEN {dec_min} AND {dec_max}
    """

    job_cat = Euclid.launch_job_async(query)
    cat = job_cat.get_results()
    print(f"  Found {len(cat)} sources")

    # Save as FITS table
    cat.write(cat_dst, format='fits', overwrite=True)
    print(f"  Wrote: {os.path.basename(cat_dst)}")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Output directory: {OUTPUT_DIR}/\n")

for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.fits*'))):
    sz_mb = os.path.getsize(f) / 1e6
    with fits.open(f) as hdul:
        dims = []
        for h in hdul:
            if h.data is not None:
                if hasattr(h, 'columns'):
                    dims.append(f"{h.name}: {len(h.data)} rows x {len(h.columns)} cols")
                else:
                    dims.append(f"{h.name} {h.data.shape}")
        info = ', '.join(dims) if dims else 'header only'
    print(f"  {os.path.basename(f):70s} {sz_mb:6.1f} MB  [{info}]")

print(f"\nCatalogue: {len(cat)} sources with positions, VIS fluxes, morphology, and flags")
print("Done.")
