"""Shared prior-parsing utilities for SHINE scene builders.

Converts :class:`~shine.config.DistributionConfig` entries (or fixed
numeric values) into NumPyro sample sites.  Supports catalog-centered
priors via the ``center="catalog"`` mechanism.
"""

from typing import Optional, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from shine.config import DistributionConfig


def parse_prior(
    name: str,
    param_config: Union[float, int, DistributionConfig],
    catalog_values: Optional[jnp.ndarray] = None,
) -> Union[float, jnp.ndarray]:
    """Create a NumPyro sample site from a config entry, or return a fixed value.

    Args:
        name: Parameter name for the NumPyro sample site.
        param_config: Either a fixed numeric value or a
            :class:`DistributionConfig` describing the prior distribution.
        catalog_values: Per-source catalog values used as the location
            parameter when ``param_config.center == "catalog"``.  Required
            when catalog-centered priors are used; ignored otherwise.

    Returns:
        Sampled value(s) from the distribution, or the fixed value.

    Raises:
        ValueError: If the distribution type is not recognized, or if
            ``center="catalog"`` is used but *catalog_values* is ``None``.
    """
    if isinstance(param_config, (float, int)):
        return float(param_config)

    catalog_centered = getattr(param_config, "center", None) == "catalog"

    if catalog_centered and catalog_values is None:
        raise ValueError(
            f"Parameter '{name}' has center='catalog' but no catalog_values "
            f"were provided"
        )

    if param_config.type == "Normal":
        if catalog_centered:
            return numpyro.sample(
                name, dist.Normal(catalog_values, param_config.sigma)
            )
        return numpyro.sample(
            name, dist.Normal(param_config.mean, param_config.sigma)
        )

    if param_config.type == "LogNormal":
        if catalog_centered:
            return numpyro.sample(
                name,
                dist.LogNormal(jnp.log(catalog_values), param_config.sigma),
            )
        return numpyro.sample(
            name,
            dist.LogNormal(jnp.log(param_config.mean), param_config.sigma),
        )

    if param_config.type == "Uniform":
        return numpyro.sample(
            name, dist.Uniform(param_config.min, param_config.max)
        )

    raise ValueError(f"Unknown distribution type: '{param_config.type}'")
