# shine.validation.extraction

Extract structured results from ArviZ InferenceData.

Provides method-aware convergence diagnostics (R-hat, ESS, divergences, BFMI)
and shear summary statistics from posterior samples. Automatically adapts
to the inference method (NUTS, MAP, or VI) via the `inference_method`
attribute on the posterior.

::: shine.validation.extraction
