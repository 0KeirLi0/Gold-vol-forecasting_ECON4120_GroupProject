"""Repeat a fixed-snapshot squared-return ARIMA selection for reproducibility."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from gold_vol_pipeline.auto_arima import hyndman_khandakar

INPUT_PATH = Path("data/processed/gold_futures.csv")
DATE_START = pd.Timestamp("2015-01-05")
DATE_END = pd.Timestamp("2026-11-02")


def main() -> None:
    """Run the same HK selection twice on a fixed local input slice."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Prepared input not found: {INPUT_PATH}. Run the gold data pipeline first."
        )
    frame = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    if not {"date", "log_return"}.issubset(frame.columns):
        raise ValueError("Input must contain 'date' and 'log_return' columns.")
    selected = frame.loc[
        frame["date"].between(DATE_START, DATE_END, inclusive="both"),
        ["date", "log_return"],
    ].dropna()
    if selected.empty:
        raise ValueError("The fixed date range contains no usable returns.")
    digest = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    print(f"Input file: {INPUT_PATH}")
    print(f"Input rows in fixed date range: {len(selected)}")
    print(
        "Date range: "
        f"{selected['date'].min().date()} through {selected['date'].max().date()}"
    )
    print(f"Input file SHA-256: {digest}")

    squared_returns = selected["log_return"].to_numpy(dtype=float) ** 2
    first = hyndman_khandakar(squared_returns)
    second = hyndman_khandakar(squared_returns)
    first_forecast = first.forecast(horizon=5)
    second_forecast = second.forecast(horizon=5)
    print(f"First selected order: ARIMA{first.order}; AICc={first.aicc:.12g}")
    print(f"Second selected order: ARIMA{second.order}; AICc={second.aicc:.12g}")
    assert first.order == second.order
    assert first.aicc == second.aicc
    assert np.array_equal(first_forecast, second_forecast)
    assert np.array_equal(first.fit.params, second.fit.params)
    print("Repeated selections and forecasts are identical.")


if __name__ == "__main__":
    main()
