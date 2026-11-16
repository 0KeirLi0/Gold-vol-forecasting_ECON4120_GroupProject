"""Out-of-sample forecast-error diagnostics for Yang-Zhang volatility models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

REQUIRED_FORECAST_COLUMNS = {
    "origin_date",
    "target_date",
    "model",
    "forecast",
    "actual",
    "converged",
}


def build_forecast_error_diagnostics(
    forecasts: pd.DataFrame,
    *,
    ljung_box_lags: Sequence[int] = (5, 10, 20),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return common-date forecast errors and per-model diagnostic summaries.

    The error convention is actual minus forecast, so a positive mean is
    average under-forecasting. Only successful forecasts on dates shared by
    every model are used in the diagnostic summaries.
    """
    missing = REQUIRED_FORECAST_COLUMNS - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast table is missing columns: {sorted(missing)}")
    if forecasts.empty:
        raise ValueError("Forecast table is empty.")

    lags = tuple(sorted(set(int(lag) for lag in ljung_box_lags)))
    if not lags or lags[0] < 1:
        raise ValueError("Ljung-Box lags must be positive integers.")

    frame = forecasts.copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    frame["forecast"] = pd.to_numeric(frame["forecast"], errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["converged"] = frame["converged"].astype("boolean").fillna(False)
    if frame[["origin_date", "target_date", "model"]].isna().any().any():
        raise ValueError("Forecast rows must have origin, target, and model values.")
    if frame.duplicated(["target_date", "model"]).any():
        raise ValueError("Forecast table contains duplicate target/model rows.")

    model_names = sorted(frame["model"].astype(str).unique())
    if len(model_names) < 1:
        raise ValueError("Forecast table contains no models.")
    successful = frame.loc[
        frame["converged"]
        & np.isfinite(frame["forecast"])
        & np.isfinite(frame["actual"])
    ].copy()
    successful["model"] = successful["model"].astype(str)
    if successful.empty:
        raise ValueError("Forecast table has no successful finite forecasts.")

    dates_by_model = {
        model: set(successful.loc[successful["model"].eq(model), "target_date"])
        for model in model_names
    }
    common_dates = set.intersection(*dates_by_model.values())
    if not common_dates:
        raise ValueError("Models have no common successful target dates.")

    errors = successful.loc[successful["target_date"].isin(common_dates)].copy()
    actual_counts = errors.groupby("target_date")["actual"].nunique()
    if actual_counts.gt(1).any():
        raise ValueError("Models report different actuals for a common target date.")
    errors["forecast_error"] = errors["actual"] - errors["forecast"]
    errors = errors.sort_values(["target_date", "model"]).reset_index(drop=True)

    summaries: list[dict[str, float | int | str]] = []
    for model in model_names:
        model_errors = errors.loc[errors["model"].eq(model), "forecast_error"]
        values = model_errors.to_numpy(dtype=float)
        if len(values) <= max(lags):
            raise ValueError(
                f"Model {model} has {len(values)} common errors, not enough for "
                f"the maximum Ljung-Box lag {max(lags)}."
            )
        row: dict[str, float | int | str] = {
            "model": model,
            "n_common": len(values),
            "mean_error": float(np.mean(values)),
            "median_error": float(np.median(values)),
            "sd_error": float(np.std(values, ddof=1)),
            "MAE": float(np.mean(np.abs(values))),
            "RMSE": float(np.sqrt(np.mean(np.square(values)))),
            "skewness": float(skew(values, bias=False)),
            "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
            "error_q05": float(np.quantile(values, 0.05)),
            "error_q95": float(np.quantile(values, 0.95)),
            "max_abs_error": float(np.max(np.abs(values))),
            "n_successful": int(
                (
                    frame["model"].astype(str).eq(model)
                    & frame["converged"]
                    & np.isfinite(frame["forecast"])
                    & np.isfinite(frame["actual"])
                ).sum()
            ),
            "failure_count": int(
                (frame["model"].astype(str).eq(model) & ~frame["converged"]).sum()
            ),
        }
        for series_name, series in (
            ("error", values),
            ("squared_error", np.square(values)),
        ):
            result = acorr_ljungbox(series, lags=list(lags), return_df=True)
            for lag in lags:
                row[f"lb_{series_name}_q_{lag}"] = float(
                    result.loc[lag, "lb_stat"]
                )
                row[f"lb_{series_name}_p_{lag}"] = float(
                    result.loc[lag, "lb_pvalue"]
                )
        summaries.append(row)

    return errors, pd.DataFrame(summaries)


def build_regime_and_extreme_diagnostics(
    errors: pd.DataFrame,
    *,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Classify realized-volatility terciles and summarize the largest misses.

    Regime labels use holdout actuals and are retrospective diagnostic groups,
    not information available to the forecast at its origin.
    """
    required = {"target_date", "model", "actual", "forecast", "forecast_error"}
    if not required <= set(errors.columns):
        raise ValueError(f"Error table is missing columns: {sorted(required - set(errors))}")
    if errors.empty:
        raise ValueError("Error table is empty.")
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    actual_by_date = errors.groupby("target_date")["actual"].agg(["first", "nunique"])
    if actual_by_date["nunique"].gt(1).any():
        raise ValueError("Models report different actuals for a target date.")
    actuals = actual_by_date["first"]
    lower, upper = (float(value) for value in actuals.quantile([1 / 3, 2 / 3]))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Realized volatility must have distinct tercile cutoffs.")

    result = errors.copy()
    result["realized_volatility_regime"] = pd.cut(
        result["actual"],
        bins=[-np.inf, lower, upper, np.inf],
        labels=["low", "middle", "high"],
        include_lowest=True,
    )
    regime_rows: list[dict[str, float | int | str]] = []
    for (model, regime), group in result.groupby(
        ["model", "realized_volatility_regime"], observed=True
    ):
        values = group["forecast_error"].to_numpy(dtype=float)
        regime_rows.append(
            {
                "model": str(model),
                "realized_volatility_regime": str(regime),
                "n": len(values),
                "mean_actual": float(group["actual"].mean()),
                "mean_error": float(np.mean(values)),
                "MAE": float(np.mean(np.abs(values))),
                "RMSE": float(np.sqrt(np.mean(np.square(values)))),
                "max_abs_error": float(np.max(np.abs(values))),
            }
        )
    regime_summary = pd.DataFrame(regime_rows)
    extremes = (
        result.assign(abs_forecast_error=result["forecast_error"].abs())
        .sort_values(["model", "abs_forecast_error"], ascending=[True, False])
        .groupby("model", sort=False)
        .head(top_n)
        .copy()
    )
    extremes["absolute_error_rank"] = extremes.groupby("model").cumcount() + 1
    extremes = extremes[
        [
            "model",
            "absolute_error_rank",
            "target_date",
            "actual",
            "forecast",
            "forecast_error",
            "abs_forecast_error",
            "realized_volatility_regime",
        ]
    ].reset_index(drop=True)
    return (
        result,
        regime_summary,
        extremes,
        {"low_upper_bound": lower, "middle_upper_bound": upper},
    )


def create_forecast_error_figures(
    errors: pd.DataFrame,
    output_dir: str,
    *,
    acf_lags: int = 20,
    regime_summary: pd.DataFrame | None = None,
) -> list[str]:
    """Write time-series, forecast-vs-error, and error-ACF figures."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    required = {"target_date", "model", "forecast", "forecast_error"}
    if not required <= set(errors.columns):
        raise ValueError(f"Error table is missing columns: {sorted(required - set(errors))}")
    if errors.empty:
        raise ValueError("Error table is empty.")
    if acf_lags < 1:
        raise ValueError("acf_lags must be positive.")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    models = sorted(errors["model"].astype(str).unique())
    paths: list[str] = []
    columns = 2
    rows = (len(models) + columns - 1) // columns

    figure, axes = plt.subplots(
        rows, columns, figsize=(12, max(3.0, rows * 2.6)), constrained_layout=True
    )
    for axis, model in zip(np.asarray(axes).reshape(-1), models):
        data = errors.loc[errors["model"].eq(model)].sort_values("target_date")
        axis.plot(data["target_date"], data["forecast_error"], linewidth=0.65)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(model.replace("_", " "))
        axis.set_ylabel("Actual − forecast")
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(alpha=0.2)
    for axis in np.asarray(axes).reshape(-1)[len(models):]:
        axis.set_visible(False)
    figure.suptitle("Out-of-sample forecast errors by target date")
    path = directory / "forecast_errors_over_time.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(
        rows, columns, figsize=(12, max(3.0, rows * 2.6)), constrained_layout=True
    )
    for axis, model in zip(np.asarray(axes).reshape(-1), models):
        data = errors.loc[errors["model"].eq(model)]
        axis.scatter(data["forecast"], data["forecast_error"], s=8, alpha=0.45)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(model.replace("_", " "))
        axis.set_xlabel("Forecast volatility")
        axis.set_ylabel("Actual − forecast")
        axis.grid(alpha=0.2)
    for axis in np.asarray(axes).reshape(-1)[len(models):]:
        axis.set_visible(False)
    figure.suptitle("Forecast errors versus forecast level")
    path = directory / "forecast_errors_vs_forecast.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(
        rows, columns, figsize=(12, max(3.0, rows * 2.6)), constrained_layout=True
    )
    for axis, model in zip(np.asarray(axes).reshape(-1), models):
        values = errors.loc[errors["model"].eq(model), "forecast_error"].to_numpy(
            dtype=float
        )
        nlags = min(acf_lags, len(values) - 1)
        error_acf = acf(values, nlags=nlags, fft=True)[1:]
        squared_acf = acf(np.square(values), nlags=nlags, fft=True)[1:]
        positions = np.arange(1, nlags + 1)
        axis.plot(positions, error_acf, marker=".", linewidth=0.8, label="Error")
        axis.plot(
            positions,
            squared_acf,
            marker=".",
            linewidth=0.8,
            label="Squared error",
        )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(model.replace("_", " "))
        axis.set_xlabel("Lag (valid target observations)")
        axis.set_ylabel("Sample ACF")
        axis.grid(alpha=0.2)
    for axis in np.asarray(axes).reshape(-1)[len(models):]:
        axis.set_visible(False)
    axes_array = np.asarray(axes).reshape(-1)
    axes_array[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Forecast-error and squared-error autocorrelation")
    path = directory / "forecast_error_acf.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    if regime_summary is not None:
        regime_figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        regime_order = ["low", "middle", "high"]
        for model in models:
            data = (
                regime_summary.loc[regime_summary["model"].eq(model)]
                .set_index("realized_volatility_regime")
                .reindex(regime_order)
            )
            axis.plot(
                regime_order,
                data["MAE"].to_numpy(dtype=float) * 1000,
                marker="o",
                linewidth=1.2,
                label=model.replace("_", " "),
            )
        axis.set(
            title="Holdout MAE by realized-volatility tercile",
            xlabel="Retrospective realized-volatility regime",
            ylabel="MAE (× 10⁻³ daily volatility)",
        )
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8, ncol=2)
        path = directory / "forecast_error_mae_by_regime.png"
        regime_figure.savefig(path, dpi=180)
        plt.close(regime_figure)
        paths.append(str(path))
    return paths
