"""Euclid VIS instrument support for SHINE.

Provides data loading, configuration, and scene modelling for
Euclid VIS quadrant-level shear inference.
"""

from shine.euclid.config import (
    EuclidDataConfig,
    EuclidInferenceConfig,
    SourceSelectionConfig,
)

__all__ = [
    "EuclidDataConfig",
    "SourceSelectionConfig",
    "EuclidInferenceConfig",
    "EuclidPSFModel",
    "EuclidExposure",
    "EuclidDataLoader",
    "ExposureSet",
    "MultiExposureScene",
    "render_model_images",
    "plot_exposure_comparison",
]


def __getattr__(name: str):
    """Lazy-load heavy submodules to avoid importing astropy at import time."""
    _data_loader_names = {
        "EuclidPSFModel", "EuclidExposure", "EuclidDataLoader", "ExposureSet",
    }
    _scene_names = {"MultiExposureScene", "render_model_images"}

    if name in _data_loader_names:
        from shine.euclid import data_loader

        return getattr(data_loader, name)

    if name in _scene_names:
        from shine.euclid import scene

        return getattr(scene, name)

    if name == "plot_exposure_comparison":
        from shine.euclid.plots import plot_exposure_comparison

        return plot_exposure_comparison

    raise AttributeError(f"module 'shine.euclid' has no attribute {name!r}")
