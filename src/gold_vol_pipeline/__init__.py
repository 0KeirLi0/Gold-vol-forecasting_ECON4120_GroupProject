"""Gold spot data preparation for volatility forecasting."""

from .pipeline import (
    DEFAULT_HORIZON,
    DEFAULT_TARGET,
    load_spot_prices_csv,
    prepare_dataset,
)

__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_TARGET",
    "load_spot_prices_csv",
    "prepare_dataset",
]
