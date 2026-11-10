"""Gold futures data preparation for volatility forecasting."""

from .pipeline import (
    DEFAULT_HORIZON,
    DEFAULT_TARGET,
    download_futures_prices,
    load_gold_prices_csv,
    prepare_dataset,
)

__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_TARGET",
    "download_futures_prices",
    "load_gold_prices_csv",
    "prepare_dataset",
]
