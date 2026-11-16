"""Inference on sequential forecast errors on common successful dates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def diagnose_forecast_errors(
    forecasts: pd.DataFrame, *, hac_lags: int = 10, alpha: float = 0.05
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Test mean errors and forecast-linked squared errors with HAC inference."""
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("hac_lags must be a nonnegative integer.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    required = {"target_date", "model", "actual", "forecast", "converged"}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise ValueError(f"Forecast data missing columns: {missing}")
    frame = forecasts.copy()
    if frame.empty or frame["model"].isna().any() or frame["model"].eq("").any():
        raise ValueError("Forecast data must contain nonempty model names and rows.")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    if frame["target_date"].isna().any() or frame.duplicated(["model", "target_date"]).any():
        raise ValueError("Each model must have one row per nonmissing target date.")
    if not frame["converged"].isin([True, False]).all():
        raise ValueError("Forecast convergence must be true or false on every row.")
    for column in ("actual", "forecast"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    models = sorted(frame["model"].unique())
    valid = frame.loc[
        frame["converged"]
        & np.isfinite(frame["actual"])
        & np.isfinite(frame["forecast"])
    ].copy()
    if (valid["actual"] < 0).any() or (valid["forecast"] <= 0).any():
        raise ValueError("Successful forecasts require nonnegative actuals and positive forecasts.")
    common_dates = set(valid["target_date"])
    for model in models:
        common_dates.intersection_update(valid.loc[valid["model"].eq(model), "target_date"])
    if len(common_dates) <= max(hac_lags + 2, 3):
        raise ValueError("Too few common successful dates for the requested HAC lag.")
    common = valid.loc[valid["target_date"].isin(common_dates)].sort_values("target_date")
    if common.groupby("target_date")["actual"].nunique().gt(1).any():
        raise ValueError("Actual targets disagree across models on common dates.")

    rows = []
    for model in models:
        sample = common.loc[common["model"].eq(model)]
        actual = sample["actual"].to_numpy(dtype=float)
        prediction = sample["forecast"].to_numpy(dtype=float)
        errors = actual - prediction
        scale = float(np.std(prediction))
        if scale <= 0:
            raise ValueError(f"Forecast values for {model} are constant.")
        mean_fit = sm.OLS(errors, np.ones((len(errors), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags}, use_t=True
        )
        centered_square = np.square(errors - errors.mean())
        standardized_forecast = (prediction - prediction.mean()) / scale
        design = np.column_stack((np.ones(len(errors)), standardized_forecast))
        variance_fit = sm.OLS(centered_square, design).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags}, use_t=True
        )
        values = np.array([
            mean_fit.params[0], mean_fit.bse[0], mean_fit.pvalues[0],
            variance_fit.params[1], variance_fit.bse[1], variance_fit.pvalues[1],
        ])
        if not np.isfinite(values).all() or mean_fit.bse[0] <= 0 or variance_fit.bse[1] <= 0:
            raise ValueError(f"HAC inference failed for model {model}.")
        interval = mean_fit.conf_int(alpha=alpha)[0]
        rows.append({
            "model": model,
            "n_common": len(sample),
            "mean_actual_minus_forecast": mean_fit.params[0],
            "mean_hac_se": mean_fit.bse[0],
            "mean_ci_lower": interval[0],
            "mean_ci_upper": interval[1],
            "mean_p_value": mean_fit.pvalues[0],
            "squared_error_slope_per_forecast_sd": variance_fit.params[1],
            "squared_error_slope_hac_se": variance_fit.bse[1],
            "squared_error_slope_p_value": variance_fit.pvalues[1],
        })

    result = pd.DataFrame(rows)
    for prefix in ("mean", "squared_error_slope"):
        result[f"{prefix}_holm_p_value"] = multipletests(
            result[f"{prefix}_p_value"], method="holm"
        )[1]
        result[f"{prefix}_reject_{alpha:g}"] = (
            result[f"{prefix}_holm_p_value"] <= alpha
        )
    metadata: dict[str, object] = {
        "error": "actual minus forecast",
        "models": models,
        "common_observations": len(common_dates),
        "first_target_date": min(common_dates).date().isoformat(),
        "last_target_date": max(common_dates).date().isoformat(),
        "excluded_noncommon_dates": int(frame["target_date"].nunique() - len(common_dates)),
        "failed_or_nonfinite_rows": int(len(frame) - len(valid)),
        "hac_maxlags": hac_lags,
        "inference": "OLS HAC, Bartlett kernel, two-sided t reference",
        "alpha": alpha,
        "multiple_testing": "Holm separately across models for each of the two test families",
        "mean_null": "unconditional mean actual-minus-forecast error equals zero",
        "variance_null": (
            "zero linear association between squared globally centered errors "
            "and the forecast, standardized across common dates"
        ),
        "limitations": (
            "The variance test is not a general test of homoskedasticity; "
            "a changing conditional mean can affect the squared-error slope. "
            "HAC inference is approximate for dependent, overlapping targets "
            "and may be sensitive to lag choice and extreme observations."
        ),
    }
    return result, metadata
