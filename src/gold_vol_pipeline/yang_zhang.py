"""Prepare daily Yang-Zhang volatility labels from a gold OHLC series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OHLC_COLUMNS = {
    "Date": "date",
    "TRDPRC_1": "close",
    "OPEN_PRC": "open",
    "HIGH_1": "high",
    "LOW_1": "low",
}


def prepare_yang_zhang_dataset(
    source: str | Path | pd.DataFrame,
    *,
    window: int = 2,
) -> pd.DataFrame:
    """Calculate unannualized Yang-Zhang volatility over trailing OHLC windows.

    Incomplete OHLC rows are retained but are not imputed; any window containing
    an incomplete bar has a missing target. The overnight return uses the
    previous observed close. The source must contain one consistently defined
    instrument and series.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least 2 observations.")

    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"OHLC input file not found: {source_path}")
        try:
            raw = pd.read_csv(source_path)
        except (OSError, pd.errors.ParserError) as error:
            raise ValueError(f"Could not read OHLC CSV {source_path}: {error}") from error

    missing = sorted(set(OHLC_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"OHLC input is missing required columns: {missing}")

    data = raw.loc[:, list(OHLC_COLUMNS)].rename(columns=OHLC_COLUMNS).copy()
    try:
        data["date"] = pd.to_datetime(data["date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("OHLC dates must be parseable.") from error
    if data["date"].isna().any():
        raise ValueError("OHLC dates must not be missing.")
    if data["date"].dt.tz is not None:
        data["date"] = data["date"].dt.tz_localize(None)
    data["date"] = data["date"].dt.normalize()
    if data["date"].duplicated().any():
        raise ValueError("OHLC input must have at most one row per date.")
    data = data.sort_values("date").reset_index(drop=True)

    price_columns = ["open", "high", "low", "close"]
    for column in price_columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        malformed = data[column].notna() & numeric.isna()
        if malformed.any():
            raise ValueError(f"{column} must contain numeric values or missing values.")
        data[column] = numeric

    complete = data[price_columns].notna().all(axis=1)
    complete_values = data.loc[complete, price_columns].to_numpy(dtype=float)
    if not np.isfinite(complete_values).all() or (complete_values <= 0).any():
        raise ValueError("Complete OHLC bars must contain finite positive prices.")
    if (
        (data.loc[complete, "high"] < data.loc[complete, ["open", "close"]].max(axis=1)).any()
        or (data.loc[complete, "low"] > data.loc[complete, ["open", "close"]].min(axis=1)).any()
        or (data.loc[complete, "high"] < data.loc[complete, "low"]).any()
    ):
        raise ValueError("Each complete OHLC bar must satisfy low <= open/close <= high.")

    overnight = np.log(data["open"] / data["close"].shift(1))
    open_to_close = np.log(data["close"] / data["open"])
    rogers_satchell = (
        np.log(data["high"] / data["open"]) * np.log(data["high"] / data["close"])
        + np.log(data["low"] / data["open"]) * np.log(data["low"] / data["close"])
    )
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    overnight_variance = overnight.rolling(window, min_periods=window).var(ddof=1)
    open_close_variance = open_to_close.rolling(window, min_periods=window).var(ddof=1)
    range_variance = rogers_satchell.rolling(window, min_periods=window).mean()
    yz_variance = (
        overnight_variance
        + k * open_close_variance
        + (1 - k) * range_variance
    )
    materially_negative = yz_variance < -1e-12
    if materially_negative.any():
        raise ValueError("Yang-Zhang variance was negative for a complete window.")
    yz_variance = yz_variance.clip(lower=0)

    data["overnight_log_return"] = overnight
    data["open_close_log_return"] = open_to_close
    data["rogers_satchell_variance"] = rogers_satchell
    data["close_return"] = np.log(data["close"]).diff()
    data["high_low_log_range"] = np.log(data["high"] / data["low"])
    data["yang_zhang_variance"] = yz_variance
    data["daily_volatility"] = np.sqrt(yz_variance)
    data["window_start_date"] = data["date"].shift(window)
    data["window_end_date"] = data["date"].where(yz_variance.notna())
    data["complete_ohlc"] = complete
    return data


def add_chronological_split(
    data: pd.DataFrame, *, train_fraction: float = 0.8
) -> pd.DataFrame:
    """Label available target rows as chronological train or test observations."""
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one.")
    if "daily_volatility" not in data:
        raise ValueError("data must include the daily_volatility target.")

    result = data.copy()
    result["split"] = "unavailable"
    valid_positions = np.flatnonzero(result["daily_volatility"].notna().to_numpy())
    if len(valid_positions) < 2:
        raise ValueError("At least 2 valid Yang-Zhang targets are required for a split.")
    train_count = int(np.floor(len(valid_positions) * train_fraction))
    if train_count < 1 or train_count >= len(valid_positions):
        raise ValueError("The available target rows are too few for the requested split.")
    result.loc[result.index[valid_positions[:train_count]], "split"] = "train"
    result.loc[result.index[valid_positions[train_count:]], "split"] = "test"
    return result
