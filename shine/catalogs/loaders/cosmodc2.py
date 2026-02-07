"""CosmoDC2 catalog loader for LSST DESC simulations.

This module loads pre-extracted CosmoDC2 catalogs from FITS or HDF5 files
and converts them to standardized GalaxyProperties for rendering.

CosmoDC2 is the LSST DESC standard simulation catalog covering 440 sq. deg.
with realistic galaxy populations, spatial clustering, and multi-band photometry.

References:
    - Data portal: https://data.lsstdesc.org/doc/cosmodc2
    - Paper: Mao et al., ApJS 234, 36 (2018), arXiv:1907.06530
"""

from pathlib import Path
from typing import Optional

import h5py
import jax.numpy as jnp
import numpy as np
from astropy.table import Table

from shine.catalogs.base import CatalogLoader, GalaxyProperties, SkyRegion
from shine.catalogs.utils import (
    compute_ellipticity,
    magnitude_to_flux,
    ra_dec_to_pixels,
)


class CosmoDC2Loader(CatalogLoader):
    """Loader for pre-extracted CosmoDC2 catalogs.

    CosmoDC2 provides single-component morphology with size, ellipticity,
    and Sersic index. Users should extract small regions from the full catalog
    at NERSC and load them locally as FITS or HDF5 files.

    Attributes:
        catalog_data: Loaded catalog table (astropy Table or dict from HDF5).
        file_format: Format of the loaded file ('fits' or 'hdf5').
    """

    def __init__(self):
        """Initialize the CosmoDC2Loader."""
        self.catalog_data: Optional[Table] = None
        self.file_format: Optional[str] = None

    def load(self, path: str) -> None:
        """Load CosmoDC2 catalog from FITS or HDF5 file.

        Args:
            path: Path to catalog file (FITS or HDF5).

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not supported.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        # Determine file format from extension
        suffix = path_obj.suffix.lower()

        if suffix in [".fits", ".fit"]:
            self.file_format = "fits"
            self.catalog_data = Table.read(path, format="fits")
        elif suffix in [".hdf5", ".h5"]:
            self.file_format = "hdf5"
            self.catalog_data = self._load_hdf5(path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                "Supported formats: .fits, .hdf5"
            )

        print(f"Loaded CosmoDC2 catalog with {len(self.catalog_data)} galaxies")

    def _load_hdf5(self, path: str) -> dict:
        """Load HDF5 file into dictionary.

        Args:
            path: Path to HDF5 file.

        Returns:
            Dictionary with column names as keys and numpy arrays as values.
        """
        data = {}
        with h5py.File(path, "r") as f:
            # Assume data is in the root group or a 'data' group
            group = f.get("data", f)
            for key in group.keys():
                data[key] = group[key][:]
        return data

    def sample_region(
        self,
        sky_region: SkyRegion,
        magnitude_limit: Optional[float] = None,
    ) -> GalaxyProperties:
        """Sample galaxies within a sky region from CosmoDC2.

        Args:
            sky_region: Sky region to sample from.
            magnitude_limit: Optional i-band magnitude cut (e.g., 25.0).

        Returns:
            GalaxyProperties with galaxies in the specified region.

        Raises:
            RuntimeError: If catalog has not been loaded.
        """
        if self.catalog_data is None:
            raise RuntimeError("Catalog not loaded. Call load() first.")

        # Get RA/Dec bounds
        ra_min, ra_max, dec_min, dec_max = sky_region.get_bounds()

        # Get catalog data (handle both Table and dict formats)
        # Convert to numpy arrays to avoid JAX/astropy conflicts
        if isinstance(self.catalog_data, Table):
            ra = np.array(self.catalog_data["ra"])
            dec = np.array(self.catalog_data["dec"])
        else:
            ra = np.array(self.catalog_data["ra"])
            dec = np.array(self.catalog_data["dec"])

        # Filter by sky region
        mask = (
            (ra >= float(ra_min))
            & (ra <= float(ra_max))
            & (dec >= float(dec_min))
            & (dec <= float(dec_max))
        )

        # Apply magnitude cut if specified
        if magnitude_limit is not None:
            if isinstance(self.catalog_data, Table):
                mag_i = np.array(self.catalog_data["mag_i_lsst"])
            else:
                mag_i = np.array(self.catalog_data["mag_i_lsst"])
            mask = mask & (mag_i < magnitude_limit)

        # Extract selected galaxies
        n_selected = np.sum(mask)
        if n_selected == 0:
            raise ValueError(
                f"No galaxies found in region "
                f"RA=[{ra_min:.4f}, {ra_max:.4f}], "
                f"Dec=[{dec_min:.4f}, {dec_max:.4f}]"
            )

        print(f"Selected {n_selected} galaxies in region")

        # Convert to GalaxyProperties
        return self._convert_to_galaxy_properties(mask, sky_region)

    def _convert_to_galaxy_properties(
        self,
        mask: np.ndarray,
        sky_region: SkyRegion,
    ) -> GalaxyProperties:
        """Convert CosmoDC2 catalog entries to GalaxyProperties.

        Args:
            mask: Boolean mask for selected galaxies.
            sky_region: Sky region for coordinate conversion.

        Returns:
            GalaxyProperties with standardized format.
        """
        # Helper to get column data
        def get_column(name: str) -> np.ndarray:
            if isinstance(self.catalog_data, Table):
                return np.array(self.catalog_data[name][mask])
            else:
                return self.catalog_data[name][mask]

        # Extract positions
        ra = get_column("ra")
        dec = get_column("dec")

        # Convert RA/Dec to pixel coordinates (placeholder - needs image size)
        # This will be properly computed in sample_postage_stamp
        x_pixels = np.zeros_like(ra)
        y_pixels = np.zeros_like(dec)

        # Extract magnitudes and convert to flux
        try:
            mag_i = get_column("mag_i_lsst")
        except (KeyError, ValueError):
            # Fallback to mag_r if mag_i not available
            mag_i = get_column("mag_r_lsst")

        flux = magnitude_to_flux(mag_i, zeropoint=30.0)

        # Extract morphology: size (HLR in arcsec)
        try:
            hlr = get_column("size")  # Major axis half-light radius
        except (KeyError, ValueError):
            # Fallback if size not available
            hlr = np.ones_like(flux) * 1.0

        # Extract ellipticity
        try:
            # CosmoDC2 provides ellipticity_1, ellipticity_2 directly
            e1 = get_column("ellipticity_1_true")
            e2 = get_column("ellipticity_2_true")
        except (KeyError, ValueError):
            # Fallback: compute from axis ratio and position angle
            try:
                size_minor = get_column("size_minor")
                position_angle = get_column("position_angle")
                e1, e2 = compute_ellipticity(hlr, size_minor, position_angle)
            except (KeyError, ValueError):
                # Last resort: assume circular
                e1 = np.zeros_like(flux)
                e2 = np.zeros_like(flux)

        # Extract redshift (optional)
        try:
            redshift = get_column("redshift")
        except (KeyError, ValueError):
            redshift = None

        # Store RA/Dec for coordinate conversion in sample_postage_stamp
        # We'll convert them properly there with image size info
        self._temp_ra = ra
        self._temp_dec = dec

        # Create GalaxyProperties
        return GalaxyProperties(
            x=jnp.array(x_pixels),
            y=jnp.array(y_pixels),
            flux=jnp.array(flux),
            half_light_radius=jnp.array(hlr),
            e1=jnp.array(e1),
            e2=jnp.array(e2),
            z=jnp.array(redshift) if redshift is not None else None,
        )

    def sample_postage_stamp(
        self,
        center_ra: float,
        center_dec: float,
        size_arcmin: float,
        pixel_scale: float,
        image_size: tuple[int, int],
        magnitude_limit: Optional[float] = None,
    ) -> GalaxyProperties:
        """Sample galaxies for a postage stamp image.

        This overrides the base class method to properly handle coordinate conversion.

        Args:
            center_ra: Center RA in degrees.
            center_dec: Center Dec in degrees.
            size_arcmin: Field size in arcminutes.
            pixel_scale: Pixel scale in arcsec/pixel.
            image_size: Image dimensions (width, height) in pixels.
            magnitude_limit: Optional magnitude cut.

        Returns:
            GalaxyProperties with pixel coordinates.
        """
        # Define sky region
        sky_region = SkyRegion(
            center_ra=center_ra,
            center_dec=center_dec,
            size_arcmin=size_arcmin,
        )

        # Sample galaxies
        galaxies = self.sample_region(sky_region, magnitude_limit)

        # Convert RA/Dec to pixels
        x_pixels, y_pixels = ra_dec_to_pixels(
            ra=self._temp_ra,
            dec=self._temp_dec,
            center_ra=center_ra,
            center_dec=center_dec,
            pixel_scale=pixel_scale,
            image_size=image_size,
        )

        # Update pixel coordinates
        galaxies = GalaxyProperties(
            x=jnp.array(x_pixels),
            y=jnp.array(y_pixels),
            flux=galaxies.flux,
            half_light_radius=galaxies.half_light_radius,
            e1=galaxies.e1,
            e2=galaxies.e2,
            z=galaxies.z,
            g1_true=galaxies.g1_true,
            g2_true=galaxies.g2_true,
        )

        # Clean up temporary storage
        del self._temp_ra
        del self._temp_dec

        return galaxies
