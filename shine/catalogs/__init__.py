"""Catalog loading and galaxy property management.

This module provides an abstraction layer for loading galaxy catalogs from
various sources (CosmoDC2, CatSim, OpenUniverse2024, Flagship2) and converting
them to a standardized format for rendering.
"""

from typing import Dict, Type

from shine.catalogs.base import CatalogLoader, GalaxyProperties, SkyRegion
from shine.catalogs.loaders.cosmodc2 import CosmoDC2Loader

# Registry of available catalog loaders
_CATALOG_REGISTRY: Dict[str, Type[CatalogLoader]] = {
    "cosmodc2": CosmoDC2Loader,
    # Additional loaders will be added as implemented:
    # "catsim": CatSimLoader,
    # "openuniverse": OpenUniverseLoader,
    # "flagship2": Flagship2Loader,
}


def register_catalog_loader(name: str, loader_class: Type[CatalogLoader]) -> None:
    """Register a catalog loader.

    Args:
        name: Catalog type identifier (e.g., "cosmodc2", "catsim").
        loader_class: CatalogLoader subclass.
    """
    _CATALOG_REGISTRY[name] = loader_class


def get_catalog_loader(catalog_type: str) -> CatalogLoader:
    """Factory function to get a catalog loader by type.

    Args:
        catalog_type: Catalog type identifier (e.g., "cosmodc2", "catsim").

    Returns:
        Instance of the appropriate CatalogLoader subclass.

    Raises:
        ValueError: If catalog type is not registered.
    """
    if catalog_type not in _CATALOG_REGISTRY:
        available = ", ".join(_CATALOG_REGISTRY.keys())
        raise ValueError(
            f"Unknown catalog type: {catalog_type}. "
            f"Available types: {available}"
        )

    loader_class = _CATALOG_REGISTRY[catalog_type]
    return loader_class()


__all__ = [
    "CatalogLoader",
    "GalaxyProperties",
    "SkyRegion",
    "register_catalog_loader",
    "get_catalog_loader",
]
