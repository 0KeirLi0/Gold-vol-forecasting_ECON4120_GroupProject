"""Gold futures data preparation for volatility forecasting."""

from .pipeline import (
    DEFAULT_HORIZON,
    DEFAULT_TARGET,
    download_futures_prices,
    load_gold_prices_csv,
    prepare_dataset,
)
from .models import (
    arch1_variance_forecast,
    ar1_squared_return_forecast,
    baseline_forecasts,
    ewma_variance_forecast,
    expanding_variance_forecast,
    garch11_variance_forecast,
    ma1_squared_return_forecast,
    rolling_variance_forecast,
)

__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_TARGET",
    "download_futures_prices",
    "load_gold_prices_csv",
    "arch1_variance_forecast",
    "ar1_squared_return_forecast",
    "baseline_forecasts",
    "ewma_variance_forecast",
    "expanding_variance_forecast",
    "garch11_variance_forecast",
    "ma1_squared_return_forecast",
    "prepare_dataset",
    "rolling_variance_forecast",
]
