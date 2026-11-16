"""Prequential bias and empirical prediction intervals for volatility forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

from .forecast_error_diagnostics import diagnose_forecast_errors
from .volatility_forecasting import _qlike


def calibrate_forecast_errors(
    forecasts: pd.DataFrame,
    *,
    models: tuple[str, ...],
    window: int = 252,
    coverage: float = 0.9,
) -> pd.DataFrame:
    """Estimate forecast-relative error distributions from past forecasts only."""
    if not models or len(set(models)) != len(models):
        raise ValueError("models must contain unique model names.")
    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer of at least 3.")
    if not np.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between zero and one.")
    required = {"origin_date", "target_date", "model", "actual", "forecast", "converged"}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise ValueError(f"Forecast data missing columns: {missing}")
    frame = forecasts.loc[forecasts["model"].isin(models)].copy()
    if set(frame["model"]) != set(models):
        raise ValueError("Forecast data does not contain every requested model.")
    for column in ("origin_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if frame[["origin_date", "target_date"]].isna().any().any():
        raise ValueError("Forecast dates must not be missing.")
    if frame.duplicated(["model", "target_date"]).any():
        raise ValueError("Each model must have one forecast per target date.")
    if (frame["origin_date"] >= frame["target_date"]).any():
        raise ValueError("Each forecast origin must precede its target date.")
    if not frame["converged"].eq(True).all():
        raise ValueError("Calibration requires successful forecasts at every requested origin.")
    for column in ("actual", "forecast"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["actual", "forecast"]].to_numpy()).all():
        raise ValueError("Calibration requires finite actuals and forecasts.")
    if (frame["actual"] < 0).any() or (frame["forecast"] <= 0).any():
        raise ValueError("Actuals must be nonnegative and forecasts strictly positive.")
    if frame.groupby("target_date")["actual"].nunique().gt(1).any():
        raise ValueError("Actual targets disagree across models.")
    dates = set(frame.loc[frame["model"].eq(models[0]), "target_date"])
    for model in models[1:]:
        if set(frame.loc[frame["model"].eq(model), "target_date"]) != dates:
            raise ValueError("Models must have the same successful target dates.")
    if len(dates) <= window:
        raise ValueError("Not enough forecasts for the requested calibration window.")

    tail = (1 - coverage) / 2
    results = []
    for model in models:
        sample = frame.loc[frame["model"].eq(model)].sort_values("target_date")
        # A previous target is usable only if it was observed at the new origin.
        if (sample["target_date"].shift(1).iloc[1:] > sample["origin_date"].iloc[1:]).any():
            raise ValueError(f"Prior targets are not available by each origin for {model}.")
        relative_errors = (sample["actual"] - sample["forecast"]) / sample["forecast"]
        previous = relative_errors.shift(1)
        rolling = previous.rolling(window=window, min_periods=window)
        relative_mean = rolling.mean()
        relative_scale = rolling.std(ddof=1)
        eligible = relative_mean.notna()
        if (relative_scale.loc[eligible] <= 0).any() or not np.isfinite(
            relative_scale.loc[eligible].to_numpy()
        ).all():
            raise ValueError(f"Past relative errors have zero or invalid scale for {model}.")
        out = sample.loc[eligible, ["origin_date", "target_date", "model", "actual"]].copy()
        out["baseline_forecast"] = sample.loc[eligible, "forecast"]
        out["relative_bias"] = relative_mean.loc[eligible]
        out["relative_error_scale"] = relative_scale.loc[eligible]
        out["forecast"] = out["baseline_forecast"] * (1 + out["relative_bias"])
        if (out["forecast"] <= 0).any():
            raise ValueError(f"Bias correction produced a nonpositive forecast for {model}.")
        out["predicted_error_scale"] = out["baseline_forecast"] * out["relative_error_scale"]
        out["standardized_error"] = (
            (out["actual"] - out["forecast"]) / out["predicted_error_scale"]
        )
        out["interval_lower"] = out["baseline_forecast"] * (
            1 + rolling.quantile(tail).loc[eligible]
        )
        out["interval_upper"] = out["baseline_forecast"] * (
            1 + rolling.quantile(1 - tail).loc[eligible]
        )
        out["calibration_n"] = window
        results.append(out)
    return pd.concat(results, ignore_index=True)


def summarize_calibration(
    forecasts: pd.DataFrame, *, coverage: float = 0.9, hac_lags: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare point errors and summarize intervals and standardized innovations."""
    if not np.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between zero and one.")
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or hac_lags < 1:
        raise ValueError("hac_lags must be a positive integer.")
    base = forecasts[["target_date", "model", "actual", "baseline_forecast"]].rename(
        columns={"baseline_forecast": "forecast"}
    )
    base["model"] = base["model"] + "__baseline"
    calibrated = forecasts[["target_date", "model", "actual", "forecast"]].copy()
    calibrated["model"] = calibrated["model"] + "__relative_calibration"
    comparison = pd.concat([base, calibrated], ignore_index=True)
    comparison["converged"] = True
    mean_tests, metadata = diagnose_forecast_errors(comparison, hac_lags=hac_lags)

    rows = []
    for model, sample in forecasts.groupby("model", sort=True):
        sample = sample.sort_values("target_date")
        actual = sample["actual"].to_numpy(dtype=float)
        z = sample["standardized_error"].to_numpy(dtype=float)
        baseline = sample["baseline_forecast"].to_numpy(dtype=float)
        lower = sample["interval_lower"].to_numpy(dtype=float)
        upper = sample["interval_upper"].to_numpy(dtype=float)
        errors = actual - sample["forecast"].to_numpy(dtype=float)
        baseline_errors = actual - baseline
        interval_score = (
            upper - lower
            + 2 / (1 - coverage) * np.maximum(lower - actual, 0)
            + 2 / (1 - coverage) * np.maximum(actual - upper, 0)
        )
        x = (baseline - baseline.mean()) / baseline.std()
        mean_fit = sm.OLS(z, np.ones((len(z), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags}, use_t=True
        )
        variance_fit = sm.OLS(
            np.square(z - z.mean()), np.column_stack((np.ones(len(z)), x))
        ).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags}, use_t=True)
        lb = acorr_ljungbox(z, lags=[10], return_df=True)
        lb_squared = acorr_ljungbox(np.square(z), lags=[10], return_df=True)
        row = {
            "model": model,
            "n": len(sample),
            "baseline_mean_error": float(np.mean(baseline_errors)),
            "baseline_mae": float(np.mean(np.abs(baseline_errors))),
            "baseline_rmse": float(np.sqrt(np.mean(np.square(baseline_errors)))),
            "baseline_qlike": _qlike(actual, baseline),
            "calibrated_mean_error": float(np.mean(errors)),
            "calibrated_mae": float(np.mean(np.abs(errors))),
            "calibrated_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "calibrated_qlike": _qlike(actual, sample["forecast"].to_numpy(dtype=float)),
            "nominal_coverage": coverage,
            "empirical_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "lower_miss_rate": float(np.mean(actual < lower)),
            "upper_miss_rate": float(np.mean(actual > upper)),
            "mean_interval_width": float(np.mean(upper - lower)),
            "mean_interval_score": float(np.mean(interval_score)),
            "standardized_error_mean": float(z.mean()),
            "standardized_error_sd": float(z.std(ddof=1)),
            "standardized_error_skew": float(skew(z, bias=False)),
            "standardized_error_excess_kurtosis": float(kurtosis(z, fisher=True, bias=False)),
            "standardized_mean_p": float(mean_fit.pvalues[0]),
            "standardized_squared_slope": float(variance_fit.params[1]),
            "standardized_squared_slope_p": float(variance_fit.pvalues[1]),
            "standardized_ljung_box_10_p": float(lb["lb_pvalue"].iloc[0]),
            "standardized_squared_ljung_box_10_p": float(lb_squared["lb_pvalue"].iloc[0]),
        }
        if not np.isfinite(np.array(list(row.values())[1:], dtype=float)).all():
            raise ValueError(f"Nonfinite distribution diagnostics for {model}.")
        rows.append(row)
    summary = pd.DataFrame(rows)
    for column in ("standardized_mean_p", "standardized_squared_slope_p"):
        summary[f"{column}_holm"] = multipletests(summary[column], method="holm")[1]
    metadata.update({
        "distribution": "empirical rolling distribution of past forecast-relative errors",
        "scale": "baseline forecast multiplied by rolling sample standard deviation of relative errors",
        "interval": "equal-tailed empirical quantiles of past forecast-relative errors",
        "nominal_coverage": coverage,
        "standardized_diagnostics": (
            "HAC mean and centered squared error versus baseline forecast; "
            "Holm across models separately for each test. Ljung-Box lag 10 "
            "on standardized errors and their squares is unadjusted and descriptive."
        ),
    })
    return summary, mean_tests, comparison, metadata
