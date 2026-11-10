"""Download and prepare daily COMEX gold futures data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

TargetMeasure = Literal["realized_variance", "realized_volatility"]
DEFAULT_HORIZON = 5
DEFAULT_TARGET: TargetMeasure = "realized_variance"
YAHOO_TICKER = "GC=F"


def download_futures_prices(start: str, end: str) -> pd.DataFrame:
    """Download Yahoo Finance's GC=F continuous futures data for [start, end).

    Yahoo's end date is exclusive. ``GC=F`` is a continuous futures series, not
    spot gold or a single fixed-expiry contract; its history may reflect rolls.
    """
    import yfinance as yf

    downloaded = yf.download(
        YAHOO_TICKER,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if downloaded.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no gold futures data for {start} through "
            f"{end} ({YAHOO_TICKER}). Check the date range, network access, and "
            "Yahoo Finance availability."
        )
    if "Close" not in downloaded.columns:
        raise RuntimeError(
            f"Yahoo Finance response for {YAHOO_TICKER} has no Close column."
        )

    close = downloaded["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise RuntimeError("Yahoo Finance returned an unexpected Close shape.")
        close = close.iloc[:, 0]

    dates = pd.DatetimeIndex(pd.to_datetime(downloaded.index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)

    prices = pd.DataFrame({"date": dates.normalize(), "close": close.to_numpy()})
    prices.attrs.update(
        source="Yahoo Finance",
        instrument="COMEX Gold Futures (continuous series, GC=F)",
        currency="USD",
        price_unit="USD per troy ounce",
    )
    return prices


def load_gold_prices_csv(
    path: str | Path,
    *,
    date_column: str = "date",
    close_column: str = "close",
    source: str = "User-supplied CSV",
    instrument: str = "Gold futures (specify contract or series)",
    currency: str = "USD",
    price_unit: str = "USD per troy ounce",
) -> pd.DataFrame:
    """Load gold prices from a CSV with explicitly selected date and close columns."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Gold-price CSV not found: {csv_path}")

    try:
        raw = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read gold-price CSV {csv_path}: {error}") from error

    missing_columns = {date_column, close_column} - set(raw.columns)
    if missing_columns:
        available = ", ".join(map(str, raw.columns))
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"CSV is missing selected column(s): {missing}. "
            f"Available columns: {available}."
        )

    prices = raw.loc[:, [date_column, close_column]].rename(
        columns={date_column: "date", close_column: "close"}
    )
    try:
        prices["date"] = pd.to_datetime(prices["date"], errors="raise")
        prices["close"] = pd.to_numeric(prices["close"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"CSV columns {date_column!r} and {close_column!r} must contain "
            "parseable dates and numeric prices."
        ) from error
    if prices["date"].dt.tz is not None:
        prices["date"] = prices["date"].dt.tz_localize(None)
    prices["date"] = prices["date"].dt.normalize()
    prices.attrs.update(
        source=source,
        instrument=instrument,
        currency=currency,
        price_unit=price_unit,
    )
    return prices


def prepare_dataset(
    prices: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    target: TargetMeasure = DEFAULT_TARGET,
) -> pd.DataFrame:
    """Build a no-look-ahead baseline dataset from daily close prices.

    Each row is a forecast origin after that date's close. Its target covers
    the next ``horizon`` observed price returns. Realized variance is the sum
    of their squared log returns; realized volatility is its square root.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of observations.")
    if target not in ("realized_variance", "realized_volatility"):
        raise ValueError(
            "target must be 'realized_variance' or 'realized_volatility'."
        )
    missing_columns = {"date", "close"} - set(prices.columns)
    if missing_columns:
        raise ValueError(
            f"prices must contain date and close columns; missing: "
            f"{', '.join(sorted(missing_columns))}."
        )

    data = prices.loc[:, ["date", "close"]].copy()
    dates = pd.to_datetime(data["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("date values must not be missing.")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    data["date"] = dates.dt.normalize()
    data["close"] = pd.to_numeric(data["close"], errors="raise")
    if data["close"].isna().any() or not np.isfinite(data["close"]).all():
        raise ValueError("close values must be finite and not missing.")
    if (data["close"] <= 0).any():
        raise ValueError("close values must be positive.")
    if data["date"].duplicated().any():
        raise ValueError("prices must contain at most one close per date.")

    data = data.sort_values("date").reset_index(drop=True)
    log_returns = np.log(data["close"]).diff()
    future_squared_returns = log_returns.pow(2).shift(-1)
    realized_variance = (
        future_squared_returns.iloc[::-1]
        .rolling(window=horizon, min_periods=horizon)
        .sum()
        .iloc[::-1]
    )

    result = pd.DataFrame(
        {
            "date": data["date"],
            "close": data["close"],
            "log_return": log_returns,
            "target_start_date": data["date"].shift(-1),
            "target_end_date": data["date"].shift(-horizon),
            target: (
                np.sqrt(realized_variance)
                if target == "realized_volatility"
                else realized_variance
            ),
        }
    )
    for key in ("source", "instrument", "currency", "price_unit"):
        if key in prices.attrs:
            result[key] = prices.attrs[key]
    result = result.dropna().reset_index(drop=True)
    if result.empty:
        raise ValueError(
            f"At least {horizon + 2} dated prices are required to calculate "
            "returns and a complete forward target."
        )
    return result
