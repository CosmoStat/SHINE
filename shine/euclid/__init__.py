"""Euclid VIS instrument support for SHINE.

Provides data loading, configuration, and scene modelling for
Euclid VIS quadrant-level shear inference.
"""

from shine.euclid.config import (
    EuclidDataConfig,
    EuclidInferenceConfig,
    PriorConfig,
    SourceSelectionConfig,
)

__all__ = [
    "EuclidDataConfig",
    "SourceSelectionConfig",
    "PriorConfig",
    "EuclidInferenceConfig",
    "EuclidPSFModel",
    "EuclidExposure",
    "EuclidDataLoader",
    "ExposureSet",
    "MultiExposureScene",
]


def __getattr__(name: str):
    """Lazy-load heavy submodules to avoid importing astropy at import time."""
    if name in ("EuclidPSFModel", "EuclidExposure", "EuclidDataLoader", "ExposureSet"):
        from shine.euclid.data_loader import (
            EuclidDataLoader,
            EuclidExposure,
            EuclidPSFModel,
            ExposureSet,
        )

        _lazy = {
            "EuclidPSFModel": EuclidPSFModel,
            "EuclidExposure": EuclidExposure,
            "EuclidDataLoader": EuclidDataLoader,
            "ExposureSet": ExposureSet,
        }
        return _lazy[name]

    if name == "MultiExposureScene":
        from shine.euclid.scene import MultiExposureScene

        return MultiExposureScene

    raise AttributeError(f"module 'shine.euclid' has no attribute {name!r}")
