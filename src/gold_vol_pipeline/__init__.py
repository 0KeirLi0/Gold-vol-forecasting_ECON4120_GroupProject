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
    generate_prediction_chart,
    main,
    ma1_squared_return_forecast,
    prediction_history,
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
    "generate_prediction_chart",
    "main",
    "ma1_squared_return_forecast",
    "prepare_dataset",
    "prediction_history",
    "rolling_variance_forecast",
]
