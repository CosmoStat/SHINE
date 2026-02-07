from shine._version import __version__

import warnings

warnings.filterwarnings(
    "ignore",
    message="Explicitly requested dtype float64",
    category=UserWarning,
)
