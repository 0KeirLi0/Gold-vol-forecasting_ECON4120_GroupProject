"""Pre-model and holdout diagnostics for the Group 2 Yang-Zhang report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, pacf

from .forecast_diagnostics import (
    MODEL_DISPLAY_NAMES,
    REQUIRED_FORECAST_COLUMNS,
    build_forecast_error_diagnostics,
)

SHORTLIST_MODELS = (
    "garch11",
    "egarch",
    "garchx_ohlc_range",
    "egarchx",
    "harx_ohlc",
    "hist_gradient_boosting_ohlc",
)

SHORTLIST_DISPLAY_NAMES = {
    **MODEL_DISPLAY_NAMES,
    "egarchx": "EGARCH-X",
    "hist_gradient_boosting_ohlc": "HGB (OHLC)",
}

DEFAULT_BASELINE_FORECASTS = Path("data/processed/stage2_yz_forecasts.csv")
DEFAULT_EGARCHX_FORECASTS = Path(
    "data/processed/stage2_yz_egarchx/stage2_yz_forecasts.csv"
)
DEFAULT_TARGET = Path("data/processed/GCc1_yang_zhang_volatility.csv")


def shortlist_display_name(model: str) -> str:
    """Return the report label for a shortlisted model code."""
    return SHORTLIST_DISPLAY_NAMES.get(model, model.replace("_", " "))


def dickey_fuller_stationary_alternative(
    series: Sequence[float] | pd.Series,
    *,
    name: str,
    sample: str,
    regression: str = "c",
    autolag: str = "AIC",
) -> dict[str, object]:
    """Run ADF with the stationary alternative and default AIC lag selection.

    ``statsmodels.tsa.stattools.adfuller`` tests a unit-root null against a
    stationary alternative. ``autolag='AIC'`` is the statsmodels default.
    """
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.ndim != 1 or values.size < 20:
        raise ValueError(f"{name} ({sample}) needs at least 20 finite observations.")
    if np.unique(values).size < 2:
        raise ValueError(f"{name} ({sample}) must have positive variance.")
    if regression not in {"c", "ct", "ctt", "n"}:
        raise ValueError("regression must be one of 'c', 'ct', 'ctt', or 'n'.")
    if autolag not in {"AIC", "BIC", "t-stat", None}:
        raise ValueError("autolag must be AIC, BIC, t-stat, or None.")

    statistic, p_value, used_lag, nobs, critical, icbest = adfuller(
        values,
        regression=regression,
        autolag=autolag,
    )
    return {
        "series": name,
        "sample": sample,
        "n_observations": int(nobs),
        "regression": regression,
        "autolag": "AIC" if autolag is None else str(autolag),
        "alternative": "stationary",
        "used_lag": int(used_lag),
        "adf_statistic": float(statistic),
        "p_value": float(p_value),
        "critical_1pct": float(critical["1%"]),
        "critical_5pct": float(critical["5%"]),
        "critical_10pct": float(critical["10%"]),
        "selected_ic": float(icbest) if np.isfinite(icbest) else np.nan,
        "reject_unit_root_5pct": bool(p_value <= 0.05),
    }


def acf_pacf_table(
    series: Sequence[float] | pd.Series,
    *,
    name: str,
    sample: str,
    nlags: int = 30,
) -> pd.DataFrame:
    """Return sample ACF and PACF with Bartlett 95% bands for the ACF."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= nlags + 2:
        raise ValueError(f"{name} ({sample}) is too short for {nlags} lags.")
    acf_values = acf(values, nlags=nlags, fft=True, alpha=0.05)
    if isinstance(acf_values, tuple):
        acf_point, acf_interval = acf_values
    else:
        acf_point, acf_interval = acf_values, None
    pacf_point = pacf(values, nlags=nlags, method="ywm")
    rows = []
    for lag in range(nlags + 1):
        row = {
            "series": name,
            "sample": sample,
            "lag": lag,
            "acf": float(acf_point[lag]),
            "pacf": float(pacf_point[lag]),
        }
        if acf_interval is not None:
            row["acf_ci_lower"] = float(acf_interval[lag, 0])
            row["acf_ci_upper"] = float(acf_interval[lag, 1])
        rows.append(row)
    return pd.DataFrame(rows)


def load_shortlist_forecasts(
    *,
    baseline_path: Path = DEFAULT_BASELINE_FORECASTS,
    egarchx_path: Path = DEFAULT_EGARCHX_FORECASTS,
    models: Sequence[str] = SHORTLIST_MODELS,
) -> pd.DataFrame:
    """Assemble the report shortlist from the saved sequential-forecast files.

    GARCH, EGARCH, range GARCH-X, HAR-X, and HGB come from the no-external
    Stage 2 run. EGARCH-X comes from the selected-predictor EGARCH-X run.
    """
    model_names = tuple(models)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("models must contain unique model names.")
    unknown = sorted(set(model_names) - set(SHORTLIST_MODELS))
    if unknown:
        raise ValueError(f"Unsupported shortlist models: {unknown}")

    frames: list[pd.DataFrame] = []
    baseline_models = [name for name in model_names if name != "egarchx"]
    if baseline_models:
        frames.append(_read_forecast_file(baseline_path, baseline_models))
    if "egarchx" in model_names:
        frames.append(_read_forecast_file(egarchx_path, ("egarchx",)))
    combined = pd.concat(frames, ignore_index=True)
    combined["model"] = pd.Categorical(
        combined["model"], categories=list(model_names), ordered=True
    )
    return combined.sort_values(["target_date", "model"]).reset_index(drop=True)


def shortlist_holdout_diagnostics(
    forecasts: pd.DataFrame,
    *,
    models: Sequence[str] = SHORTLIST_MODELS,
    ljung_box_lags: Sequence[int] = (5, 10, 20),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return common-date holdout errors and Ljung-Box summaries."""
    return build_forecast_error_diagnostics(
        forecasts, models=models, ljung_box_lags=ljung_box_lags
    )


def _read_forecast_file(path: Path, models: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Forecast file not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_FORECAST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    frame["model"] = frame["model"].astype(str)
    selected = frame.loc[frame["model"].isin(models)].copy()
    present = set(selected["model"].unique())
    missing_models = sorted(set(models) - present)
    if missing_models:
        raise ValueError(f"{path} is missing models: {missing_models}")
    return selected


def load_prepared_target(path: Path = DEFAULT_TARGET) -> pd.DataFrame:
    """Load the prepared GCc1 Yang-Zhang file used by the report."""
    if not path.is_file():
        raise FileNotFoundError(f"Prepared target not found: {path}")
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "close", "daily_volatility", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame.sort_values("date").reset_index(drop=True)


def assign_forecast_quintiles(forecasts: pd.Series, *, quantiles: int = 5) -> pd.Series:
    """Assign equal-count forecast bins, matching the report Q5 definition."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")
    values = pd.to_numeric(forecasts, errors="raise")
    if values.isna().any():
        raise ValueError("Forecasts used for quintiles must be finite.")
    if len(values) < quantiles:
        raise ValueError("Need at least one observation per quantile.")
    ordered = values.sort_values(kind="mergesort")
    labels = np.asarray(
        pd.qcut(
            np.arange(len(ordered)), q=quantiles, labels=False, duplicates="drop"
        ),
        dtype=int,
    ) + 1
    result = pd.Series(index=forecasts.index, dtype=int)
    result.loc[ordered.index] = labels
    return result


def q5_influence_table(
    errors: pd.DataFrame,
    *,
    k_values: Sequence[int] = (1, 2, 5),
) -> pd.DataFrame:
    """Measure how much forecast-Q5 variance is carried by the largest misses.

    For each model, Q5 is the top fifth of that model's forecasts. The table
    reports original Q5 variance, the share of Q5 sum of squared errors in the
    k largest absolute errors, and Q5 variance after dropping those k dates
    from the original Q5 set.
    """
    required = {"model", "target_date", "forecast", "actual", "forecast_error"}
    missing = sorted(required - set(errors.columns))
    if missing:
        raise ValueError(f"Error table is missing columns: {missing}")
    if errors.empty:
        raise ValueError("Error table is empty.")
    ks = tuple(int(k) for k in k_values)
    if not ks or any(k < 1 for k in ks):
        raise ValueError("k_values must be positive integers.")

    rows = []
    for model, group in errors.groupby("model", sort=False):
        frame = group.dropna(subset=["forecast", "actual", "forecast_error"]).copy()
        frame["forecast_quintile"] = assign_forecast_quintiles(frame["forecast"])
        q5 = frame.loc[frame["forecast_quintile"].eq(5)].copy()
        q1 = frame.loc[frame["forecast_quintile"].eq(1)]
        if len(q5) < 2 or len(q1) < 2:
            raise ValueError(f"{model} does not have enough Q1/Q5 observations.")
        q5_var = float(np.var(q5["forecast_error"], ddof=1))
        q1_var = float(np.var(q1["forecast_error"], ddof=1))
        sse = float(np.sum(np.square(q5["forecast_error"])))
        ranked = q5.reindex(q5["forecast_error"].abs().sort_values(ascending=False).index)
        for k in ks:
            if k >= len(q5):
                raise ValueError(f"{model} Q5 has fewer than {k} observations.")
            dropped = ranked.iloc[:k]
            kept = ranked.iloc[k:]
            kept_var = float(np.var(kept["forecast_error"], ddof=1))
            dropped_sse = float(np.sum(np.square(dropped["forecast_error"])))
            rows.append(
                {
                    "model": str(model),
                    "k_dropped": k,
                    "n_q5": int(len(q5)),
                    "q5_variance": q5_var,
                    "q1_variance": q1_var,
                    "q5_q1_ratio": q5_var / q1_var if q1_var > 0 else np.nan,
                    "dropped_dates": ",".join(
                        pd.to_datetime(dropped["target_date"]).dt.strftime("%Y-%m-%d")
                    ),
                    "dropped_sse_share_of_q5": dropped_sse / sse if sse > 0 else np.nan,
                    "q5_variance_after_drop": kept_var,
                    "q5_variance_decline": (
                        1.0 - kept_var / q5_var if q5_var > 0 else np.nan
                    ),
                    "q5_q1_ratio_after_drop": (
                        kept_var / q1_var if q1_var > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def subperiod_metrics(
    errors: pd.DataFrame,
    *,
    splits: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    """Compute MAE, RMSE, and forecast-Q5/Q1 variance by named date windows."""
    required = {"model", "target_date", "forecast", "actual", "forecast_error"}
    missing = sorted(required - set(errors.columns))
    if missing:
        raise ValueError(f"Error table is missing columns: {missing}")
    if not splits:
        raise ValueError("splits must contain at least one named window.")
    frame = errors.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    rows = []
    for name, (start, end) in splits.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        window = frame.loc[frame["target_date"].between(start_ts, end_ts)]
        if window.empty:
            raise ValueError(f"Split {name} contains no rows.")
        for model, group in window.groupby("model", sort=False):
            err = group["forecast_error"].to_numpy(dtype=float)
            group = group.copy()
            group["forecast_quintile"] = assign_forecast_quintiles(group["forecast"])
            q5 = group.loc[group["forecast_quintile"].eq(5), "forecast_error"]
            q1 = group.loc[group["forecast_quintile"].eq(1), "forecast_error"]
            rows.append(
                {
                    "split": name,
                    "start": start_ts.strftime("%Y-%m-%d"),
                    "end": end_ts.strftime("%Y-%m-%d"),
                    "model": str(model),
                    "n": int(len(group)),
                    "MAE": float(np.mean(np.abs(err))),
                    "RMSE": float(np.sqrt(np.mean(np.square(err)))),
                    "mean_error": float(np.mean(err)),
                    "q5_variance": float(np.var(q5, ddof=1)) if len(q5) > 1 else np.nan,
                    "q1_variance": float(np.var(q1, ddof=1)) if len(q1) > 1 else np.nan,
                    "q5_q1_ratio": (
                        float(np.var(q5, ddof=1) / np.var(q1, ddof=1))
                        if len(q5) > 1 and len(q1) > 1
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def train_fraction_for_cutoff(
    dates: pd.Series | Sequence[object],
    cutoff: str | pd.Timestamp,
) -> float:
    """Return a train fraction whose floor split ends on ``cutoff``.

    ``evaluate_daily_volatility`` uses ``floor(n * train_fraction)``. Adding 0.5
    observations keeps that floor equal to the count of dates on or before the
    cutoff even if ``n * (train / n)`` rounds down in floating point.
    """
    parsed = pd.to_datetime(pd.Series(dates), errors="raise")
    if parsed.isna().any():
        raise ValueError("Cutoff alignment dates must be parseable.")
    cutoff_ts = pd.Timestamp(cutoff)
    n = int(len(parsed))
    train = int((parsed <= cutoff_ts).sum())
    if train < 1 or train >= n:
        raise ValueError("Cutoff does not leave both training and test observations.")
    return (train + 0.5) / n


def month_dummy_matrix(dates: pd.Series | Sequence[object]) -> np.ndarray:
    """Eleven calendar-month dummies with January as the omitted base month."""
    months = pd.to_datetime(pd.Series(dates), errors="raise").dt.month.to_numpy()
    if np.isnan(months.astype(float)).any():
        raise ValueError("Month dummy dates must be parseable.")
    dummies = np.zeros((len(months), 11), dtype=float)
    for index, month in enumerate(months):
        month_i = int(month)
        if month_i != 1:
            dummies[index, month_i - 2] = 1.0
    return dummies


def forecast_harx_with_month_dummies(
    history: pd.DataFrame,
    target_date: str | pd.Timestamp,
) -> float:
    """HAR-X OLS on log volatility plus month dummies of the target session."""
    from .volatility_forecasting import HAR_LAGS, _lagged_design

    if "date" not in history.columns:
        raise ValueError("history must contain a date column.")
    features, targets, forecast_features = _lagged_design(history, include_ohlc=True)
    history_dates = pd.to_datetime(history["date"], errors="raise")
    train_dates = history_dates.iloc[HAR_LAGS : HAR_LAGS + len(features)]
    if len(train_dates) != len(features):
        raise ValueError("HAR month-dummy dates do not match the lagged design.")
    design = np.column_stack(
        (np.ones(len(features)), features, month_dummy_matrix(train_dates))
    )
    forecast_design = np.column_stack(
        (
            np.ones(1),
            forecast_features,
            month_dummy_matrix(pd.Series([pd.Timestamp(target_date)])),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    prediction = float((forecast_design @ coefficients).item())
    value = float(np.exp(prediction))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("HAR-X month-dummy forecast is non-positive or non-finite.")
    return value


def evaluate_harx_month_dummies(
    data: pd.DataFrame,
    *,
    last_train_date: str | pd.Timestamp = "2023-06-01",
) -> pd.DataFrame:
    """Sequential one-step HAR-X forecasts with next-session month dummies."""
    required = {
        "date",
        "daily_volatility",
        "close_return",
        "overnight_log_return",
        "open_close_log_return",
        "high_low_log_range",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").dropna(subset=["daily_volatility"]).reset_index(
        drop=True
    )
    cutoff = pd.Timestamp(last_train_date)
    split_position = int((frame["date"] <= cutoff).sum())
    if split_position < 30 or split_position >= len(frame):
        raise ValueError("Month-dummy split leaves too few training or test rows.")
    rows: list[dict[str, object]] = []
    for test_position in range(split_position, len(frame)):
        history = frame.iloc[:test_position].copy()
        target_date = pd.Timestamp(frame.at[test_position, "date"])
        actual = float(frame.at[test_position, "daily_volatility"])
        origin = pd.Timestamp(history.iloc[-1]["date"])
        try:
            forecast = forecast_harx_with_month_dummies(history, target_date)
            error = ""
        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ) as exception:
            forecast = np.nan
            error = f"{type(exception).__name__}: {exception}"
        rows.append(
            {
                "origin_date": origin,
                "target_date": target_date,
                "model": "harx_month_dummies",
                "forecast": forecast,
                "actual": actual,
                "converged": not error,
                "error": error,
                "forecast_error": (
                    actual - float(forecast) if np.isfinite(forecast) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def split_series(
    target: pd.DataFrame, column: str, *, valid_only: bool = True
) -> dict[str, pd.Series]:
    """Return full, training, and test extracts of one prepared column."""
    if column not in target.columns:
        raise ValueError(f"Prepared target is missing {column}.")
    frame = target.copy()
    if valid_only:
        frame = frame.loc[frame[column].notna()]
    values = pd.to_numeric(frame[column], errors="raise")
    dates = pd.to_datetime(frame["date"], errors="raise")
    series = pd.Series(values.to_numpy(dtype=float), index=dates, name=column)
    splits: dict[str, pd.Series] = {"full": series}
    if "split" in frame.columns:
        for label in ("train", "test"):
            mask = frame["split"].eq(label)
            splits[label] = pd.Series(
                values.loc[mask].to_numpy(dtype=float),
                index=dates.loc[mask],
                name=column,
            )
    return splits
