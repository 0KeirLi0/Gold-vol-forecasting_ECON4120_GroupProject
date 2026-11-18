"""Descriptive calendar-month summaries for Yang-Zhang daily volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "daily_volatility", "split"}
SAMPLES = ("all", "train", "test")


def _summary(values: pd.Series, dates: pd.Series) -> dict[str, float | int | str]:
    numeric = values.to_numpy(dtype=float)
    return {
        "n_targets": len(numeric),
        "n_years": int(dates.dt.year.nunique()),
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
        "mean_daily_volatility": float(np.mean(numeric)),
        "median_daily_volatility": float(np.median(numeric)),
        "sd_daily_volatility": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else np.nan,
        "q25_daily_volatility": float(np.quantile(numeric, 0.25)),
        "q75_daily_volatility": float(np.quantile(numeric, 0.75)),
        "q90_daily_volatility": float(np.quantile(numeric, 0.90)),
    }


def summarize_yang_zhang_calendar(
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize valid target volatility by month-of-year and year-month.

    Both tables contain all valid target observations plus separate train and
    test summaries. Missing target labels (for incomplete OHLC windows) are
    excluded. These summaries are descriptive and do not test seasonality.
    """
    missing = REQUIRED_COLUMNS - set(targets.columns)
    if missing:
        raise ValueError(f"Target table is missing columns: {sorted(missing)}")
    if targets.empty:
        raise ValueError("Target table is empty.")

    frame = targets.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("Target dates must not be missing.")
    if frame["date"].dt.tz is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("Target table must have at most one row per date.")

    numeric = pd.to_numeric(frame["daily_volatility"], errors="coerce")
    malformed = frame["daily_volatility"].notna() & numeric.isna()
    if malformed.any():
        raise ValueError("daily_volatility must be numeric or missing.")
    frame["daily_volatility"] = numeric
    valid = frame["daily_volatility"].notna()
    values = frame.loc[valid, "daily_volatility"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Valid daily_volatility values must be finite and nonnegative.")

    frame["split"] = frame["split"].astype("string")
    if frame.loc[valid, "split"].isna().any() or not frame.loc[
        valid, "split"
    ].isin(["train", "test"]).all():
        raise ValueError("Every valid target must have split='train' or split='test'.")
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("Target table contains no valid daily-volatility values.")
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["month_name"] = frame["date"].dt.month_name()

    month_rows: list[dict[str, float | int | str]] = []
    calendar_rows: list[dict[str, float | int | str]] = []
    for sample in SAMPLES:
        sample_frame = (
            frame if sample == "all" else frame.loc[frame["split"].eq(sample)]
        )
        for month, group in sample_frame.groupby("month", sort=True):
            month_rows.append(
                {
                    "sample": sample,
                    "month": int(month),
                    "month_name": str(group["month_name"].iloc[0]),
                    **_summary(group["daily_volatility"], group["date"]),
                }
            )
        for (year, month), group in sample_frame.groupby(
            ["year", "month"], sort=True
        ):
            calendar_rows.append(
                {
                    "sample": sample,
                    "year": int(year),
                    "month": int(month),
                    "month_name": str(group["month_name"].iloc[0]),
                    **_summary(group["daily_volatility"], group["date"]),
                }
            )

    month_summary = pd.DataFrame(month_rows).sort_values(
        ["sample", "month"], key=lambda col: col.map(
            {"all": 0, "train": 1, "test": 2}
        ) if col.name == "sample" else col
    ).reset_index(drop=True)
    calendar_summary = pd.DataFrame(calendar_rows).sort_values(
        ["sample", "year", "month"], key=lambda col: col.map(
            {"all": 0, "train": 1, "test": 2}
        ) if col.name == "sample" else col
    ).reset_index(drop=True)
    return month_summary, calendar_summary
