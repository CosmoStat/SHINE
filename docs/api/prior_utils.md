# shine.prior_utils

Shared prior-parsing utilities for SHINE scene builders.

Converts `DistributionConfig` entries (or fixed numeric values) into NumPyro
sample sites. Supports catalog-centered priors via the `center="catalog"`
mechanism, where the distribution location parameter comes from per-source
catalog data at runtime.

::: shine.prior_utils
