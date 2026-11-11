"""Shared rolling-origin forecasts and out-of-sample evaluation statistics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano test result; positive differences mean model A is worse."""

    statistic: float
    p_value: float
    mean_loss_difference: float


@dataclass(frozen=True)
class MZResult:
    """Mincer-Zarnowitz calibration regression and joint Wald test."""

    intercept: float
    slope: float
    wald_p_value: float


@dataclass(frozen=True)
class ForecastContext:
    """Data available to a model at one rolling forecast origin."""

    history: pd.DataFrame
    training: pd.DataFrame
    origin: pd.Timestamp
    horizon: int


def _validated_pair(
    actual: Sequence[float] | pd.Series,
    forecast: Sequence[float] | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        actual_values = np.asarray(actual, dtype=float)
        forecast_values = np.asarray(forecast, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("actual and forecast must contain finite numeric values.") from error
    if (
        actual_values.ndim != 1
        or forecast_values.ndim != 1
        or len(actual_values) == 0
        or len(actual_values) != len(forecast_values)
    ):
        raise ValueError("actual and forecast must be non-empty, equal-length vectors.")
    if not np.isfinite(actual_values).all() or not np.isfinite(forecast_values).all():
        raise ValueError("actual and forecast must contain finite numeric values.")
    return actual_values, forecast_values


def _loss_result(losses: np.ndarray, index: pd.Index | None = None) -> tuple[pd.Series, float]:
    series = pd.Series(losses, index=index, name="loss")
    return series, float(series.mean())


def mse(
    actual: Sequence[float] | pd.Series,
    forecast: Sequence[float] | pd.Series,
) -> tuple[pd.Series, float]:
    """Return per-origin squared errors and their mean."""
    actual_values, forecast_values = _validated_pair(actual, forecast)
    index = actual.index if isinstance(actual, pd.Series) else None
    return _loss_result(np.square(actual_values - forecast_values), index)


def mae(
    actual: Sequence[float] | pd.Series,
    forecast: Sequence[float] | pd.Series,
) -> tuple[pd.Series, float]:
    """Return per-origin absolute errors and their mean."""
    actual_values, forecast_values = _validated_pair(actual, forecast)
    index = actual.index if isinstance(actual, pd.Series) else None
    return _loss_result(np.abs(actual_values - forecast_values), index)


def qlike(
    actual: Sequence[float] | pd.Series,
    forecast: Sequence[float] | pd.Series,
) -> tuple[pd.Series, float]:
    """Return Patton's robust QLIKE losses and their mean for positive variances."""
    actual_values, forecast_values = _validated_pair(actual, forecast)
    if (actual_values <= 0).any() or (forecast_values <= 0).any():
        raise ValueError("QLIKE requires strictly positive actuals and forecasts.")
    ratio = actual_values / forecast_values
    index = actual.index if isinstance(actual, pd.Series) else None
    return _loss_result(ratio - np.log(ratio) - 1, index)


def _validated_losses(
    loss: Sequence[float] | pd.Series,
    name: str,
) -> np.ndarray:
    try:
        values = np.asarray(loss, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric losses.") from error
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain at least two finite loss observations.")
    return values


def diebold_mariano(
    loss_a: Sequence[float] | pd.Series,
    loss_b: Sequence[float] | pd.Series,
    *,
    horizon: int,
) -> DMResult:
    """Compare mean losses using Newey-West HAC variance and the HLN correction.

    The reported mean is ``loss_a - loss_b``; a negative statistic favors A.
    """
    a = _validated_losses(loss_a, "loss_a")
    b = _validated_losses(loss_b, "loss_b")
    if len(a) != len(b):
        raise ValueError("loss_a and loss_b must have the same number of observations.")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    differences = a - b
    count = len(differences)
    mean_difference = float(differences.mean())
    centered = differences - mean_difference
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, min(horizon - 1, count - 1) + 1):
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1 - lag / horizon
        long_run_variance += 2 * weight * autocovariance
    long_run_variance = max(long_run_variance, 0.0)
    if long_run_variance == 0:
        statistic = 0.0 if mean_difference == 0 else float(np.sign(mean_difference) * np.inf)
    else:
        statistic = mean_difference / np.sqrt(long_run_variance / count)

    correction_squared = (
        count + 1 - 2 * horizon + horizon * (horizon - 1) / count
    ) / count
    correction = np.sqrt(max(correction_squared, 0.0))
    corrected_statistic = float(statistic * correction)
    if np.isfinite(corrected_statistic):
        p_value = float(2 * student_t.sf(abs(corrected_statistic), df=count - 1))
    else:
        p_value = 0.0
    return DMResult(corrected_statistic, p_value, mean_difference)


def mincer_zarnowitz(
    actual: Sequence[float] | pd.Series,
    forecast: Sequence[float] | pd.Series,
    *,
    horizon: int,
) -> MZResult:
    """Fit the HAC Mincer-Zarnowitz regression and test (intercept, slope)=(0,1)."""
    actual_values, forecast_values = _validated_pair(actual, forecast)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if len(actual_values) < 3:
        raise ValueError("Mincer-Zarnowitz testing requires at least three observations.")
    design = sm.add_constant(forecast_values, has_constant="add")
    fit = sm.OLS(actual_values, design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": horizon - 1},
    )
    wald = fit.wald_test(
        (np.eye(2), np.array([0.0, 1.0])),
        scalar=True,
    )
    return MZResult(float(fit.params[0]), float(fit.params[1]), float(wald.pvalue))


def forecast_combination(
    forecasts: Mapping[str, float],
    *,
    models: Sequence[str] = ("garch11", "ewma", "har"),
) -> float:
    """Return an equal-weight average, defaulting to GARCH, EWMA, and HAR."""
    if not models or len(set(models)) != len(models):
        raise ValueError("models must contain unique forecast names.")
    missing = sorted(set(models) - forecasts.keys())
    if missing:
        raise ValueError(f"forecasts are missing combination inputs: {missing}")
    try:
        values = np.asarray([forecasts[name] for name in models], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("combination forecasts must be finite numeric values.") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("combination forecasts must be finite and non-negative.")
    return float(values.mean())


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _forecast_details(
    result: object,
) -> tuple[float, float | None, str | None]:
    alpha: float | None = None
    coefficients: str | None = None
    if hasattr(result, "alpha") and hasattr(result, "coefficients"):
        alpha = float(getattr(result, "alpha"))
        raw_coefficients = getattr(result, "coefficients")
        if isinstance(raw_coefficients, pd.Series):
            coefficients = json.dumps(
                {str(key): float(value) for key, value in raw_coefficients.items()}
            )
    if hasattr(result, "forecast"):
        result = getattr(result, "forecast")
    elif isinstance(result, Mapping) and "forecast" in result:
        result = result["forecast"]
    try:
        forecast = float(result)
    except (TypeError, ValueError) as error:
        raise ValueError("model must return a numeric forecast.") from error
    if not np.isfinite(forecast) or forecast < 0:
        raise ValueError("model forecast must be finite and non-negative.")
    return forecast, alpha, coefficients


def rolling_origin_evaluation(
    data: pd.DataFrame,
    models: Mapping[str, Callable[[ForecastContext], object]],
    *,
    horizon: int,
    min_train: int = 750,
    step: int = 1,
    window: str = "expanding",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate models at common origins with target-end-date purging.

    Each callable receives a ``ForecastContext``. ``history`` contains only
    returns and predictors through the origin; ``training`` contains rows whose
    ``target_end_date`` is no later than that origin. Add ``"combination"`` to
    ``models`` to create the default GARCH/EWMA/HAR equal-weight combination.
    """
    horizon = _positive_integer(horizon, "horizon")
    min_train = _positive_integer(min_train, "min_train")
    step = _positive_integer(step, "step")
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be 'expanding' or 'rolling'.")
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a non-empty pandas DataFrame.")
    required = {"date", "log_return", "target_end_date", "realized_variance"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if not models:
        raise ValueError("models must contain at least one named callable.")
    if any(not isinstance(name, str) or not name for name in models):
        raise ValueError("model names must be non-empty strings.")
    if any(not callable(model) for model in models.values()):
        raise ValueError("each model must be callable.")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.tz_localize(None)
    frame["target_end_date"] = pd.to_datetime(
        frame["target_end_date"], errors="raise"
    ).dt.tz_localize(None)
    if frame["date"].isna().any() or frame["target_end_date"].isna().any():
        raise ValueError("date and target_end_date must not be missing.")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("data must have a unique, increasing date column.")
    frame["log_return"] = pd.to_numeric(frame["log_return"], errors="raise")
    frame["realized_variance"] = pd.to_numeric(
        frame["realized_variance"], errors="raise"
    )
    if frame["log_return"].isna().any() or not np.isfinite(frame["log_return"]).all():
        raise ValueError("log_return must contain finite values.")
    valid_actual = frame["realized_variance"].notna()
    if not np.isfinite(frame.loc[valid_actual, "realized_variance"]).all():
        raise ValueError("realized_variance must contain finite values or missing labels.")
    if (frame.loc[valid_actual, "realized_variance"] < 0).any():
        raise ValueError("realized_variance must be non-negative.")

    target_column = "realized_variance"
    model_names = list(models)
    rows: list[dict[str, object]] = []
    output_model_names = [
        name for name in model_names if name != "combination"
    ] + (["combination"] if "combination" in models else [])
    for position in range(0, len(frame), step):
        row = frame.iloc[position]
        if pd.isna(row[target_column]):
            continue
        origin = pd.Timestamp(row["date"])
        available = frame.loc[frame["date"] <= origin]
        eligible = available.loc[available["target_end_date"] <= origin]
        if window == "rolling":
            eligible = eligible.tail(min_train)
        if len(eligible) < min_train:
            continue
        history_columns = [
            column
            for column in frame.columns
            if column
            not in {
                target_column,
                "target_start_date",
                "target_end_date",
                "realized_volatility",
            }
        ]
        context = ForecastContext(
            history=available.loc[:, history_columns].copy(),
            training=eligible.copy(),
            origin=origin,
            horizon=horizon,
        )
        origin_forecasts: dict[str, float] = {}
        origin_details: dict[str, tuple[float | None, str | None]] = {}
        origin_errors: dict[str, str] = {}
        for name, model in models.items():
            if name == "combination":
                continue
            try:
                forecast, alpha, coefficients = _forecast_details(model(context))
                origin_forecasts[name] = forecast
                origin_details[name] = (alpha, coefficients)
            except (ValueError, RuntimeError, NotImplementedError) as error:
                origin_errors[name] = str(error)
        if "combination" in models:
            try:
                origin_forecasts["combination"] = forecast_combination(
                    origin_forecasts
                )
            except ValueError as error:
                origin_errors["combination"] = str(error)
        for name in output_model_names:
            converged = name in origin_forecasts
            rows.append(
                {
                    "origin_date": origin,
                    "model": name,
                    "forecast": origin_forecasts.get(name, np.nan),
                    "actual": float(row[target_column]),
                    "converged": converged,
                    "error": origin_errors.get(name, ""),
                    "selected_alpha": (
                        origin_details.get(name, (None, None))[0]
                    ),
                    "coefficients": (
                        origin_details.get(name, (None, None))[1]
                    ),
                }
            )
    if not rows:
        raise ValueError(
            "No forecast rows were generated; check min_train, target dates, "
            "and the supplied data range."
        )
    results = pd.DataFrame(rows)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path, index=False)
        package_versions = {}
        for package in ("numpy", "pandas", "scipy", "statsmodels", "arch", "scikit-learn"):
            try:
                package_versions[package] = version(package)
            except PackageNotFoundError:
                package_versions[package] = None
        manifest = {
            "data_range": {
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(),
            },
            "horizon": horizon,
            "min_train": min_train,
            "step": step,
            "window": window,
            "package_versions": package_versions,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = path.with_name(f"{path.stem}_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return results


def dm_pvalue_matrix(
    results: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    loss: str = "qlike",
    horizon: int = 5,
) -> pd.DataFrame:
    """Return pairwise DM p-values for the common origins of selected models."""
    if loss not in {"mse", "qlike"}:
        raise ValueError("loss must be 'mse' or 'qlike'.")
    if not {"origin_date", "model", "forecast", "actual"}.issubset(results.columns):
        raise ValueError("results must contain origin_date, model, forecast, and actual.")
    selected = list(models) if models is not None else sorted(results["model"].unique())
    matrix = pd.DataFrame(np.nan, index=selected, columns=selected, dtype=float)
    for model_a in selected:
        matrix.loc[model_a, model_a] = 1.0
        a = results.loc[results["model"] == model_a, ["origin_date", "forecast", "actual"]]
        for model_b in selected:
            if model_b == model_a:
                continue
            b = results.loc[results["model"] == model_b, ["origin_date", "forecast"]]
            merged = a.merge(b, on="origin_date", suffixes=("_a", "_b"))
            merged = merged.dropna()
            if len(merged) < 2:
                continue
            if loss == "mse":
                loss_a = np.square(merged["actual"] - merged["forecast_a"])
                loss_b = np.square(merged["actual"] - merged["forecast_b"])
            else:
                if (merged["actual"] <= 0).any() or (
                    (merged["forecast_a"] <= 0) | (merged["forecast_b"] <= 0)
                ).any():
                    continue
                ratio_a = merged["actual"] / merged["forecast_a"]
                ratio_b = merged["actual"] / merged["forecast_b"]
                loss_a = ratio_a - np.log(ratio_a) - 1
                loss_b = ratio_b - np.log(ratio_b) - 1
            matrix.loc[model_a, model_b] = diebold_mariano(
                loss_a, loss_b, horizon=horizon
            ).p_value
    return matrix


def summary_table(
    results: pd.DataFrame,
    benchmark: str = "garch11",
    *,
    horizon: int = 5,
    top_n: int = 5,
) -> pd.DataFrame:
    """Summarize forecast losses, benchmark DM tests, and MZ calibration."""
    required = {"origin_date", "model", "forecast", "actual"}
    if not required.issubset(results.columns):
        raise ValueError(f"results must contain columns: {sorted(required)}")
    if benchmark not in set(results["model"]):
        raise ValueError(f"benchmark model '{benchmark}' is not present in results.")
    horizon = _positive_integer(horizon, "horizon")
    top_n = _positive_integer(top_n, "top_n")

    model_results = {
        model_name: results.loc[
            (results["model"] == model_name) & results["forecast"].notna()
        ].sort_values("origin_date")
        for model_name in results["model"].unique()
    }
    if model_results[benchmark].empty:
        raise ValueError(f"benchmark model '{benchmark}' has no valid forecasts.")
    benchmark_frame = model_results[benchmark][["origin_date", "actual", "forecast"]]
    records: list[dict[str, object]] = []
    pvalue_locations: list[tuple[int, str]] = []
    pvalues: list[float] = []
    for name, model_frame in model_results.items():
        joined = model_frame[["origin_date", "actual", "forecast"]].merge(
            benchmark_frame,
            on="origin_date",
            suffixes=("", "_benchmark"),
            validate="one_to_one",
        )
        actual = model_frame["actual"].to_numpy(dtype=float)
        forecast = model_frame["forecast"].to_numpy(dtype=float)
        if len(model_frame) == 0:
            records.append(
                {
                    "model": name, "n": 0, "MSE": np.nan, "MAE": np.nan,
                    "QLIKE": np.nan, "MSE_ratio_to_benchmark": np.nan,
                    "QLIKE_ratio_to_benchmark": np.nan, "DM_MSE_p_raw": np.nan,
                    "DM_MSE_p_holm": np.nan, "DM_QLIKE_p_raw": np.nan,
                    "DM_QLIKE_p_holm": np.nan, "MZ_a": np.nan, "MZ_b": np.nan,
                    "MZ_wald_p": np.nan,
                }
            )
            continue
        _, model_mse = mse(actual, forecast)
        _, model_mae = mae(actual, forecast)
        try:
            _, model_qlike = qlike(actual, forecast)
        except ValueError:
            model_qlike = np.nan
        if len(joined) >= 2:
            mse_a = np.square(joined["actual"] - joined["forecast"])
            mse_b = np.square(joined["actual"] - joined["forecast_benchmark"])
            mse_test = diebold_mariano(mse_a, mse_b, horizon=horizon)
            if (
                (joined["actual"] > 0).all()
                and (joined["forecast"] > 0).all()
                and (joined["forecast_benchmark"] > 0).all()
            ):
                qlike_a = (
                    joined["actual"] / joined["forecast"]
                    - np.log(joined["actual"] / joined["forecast"]) - 1
                )
                qlike_b = (
                    joined["actual"] / joined["forecast_benchmark"]
                    - np.log(joined["actual"] / joined["forecast_benchmark"]) - 1
                )
                qlike_test = diebold_mariano(qlike_a, qlike_b, horizon=horizon)
            else:
                qlike_test = DMResult(np.nan, np.nan, np.nan)
            mz = mincer_zarnowitz(actual, forecast, horizon=horizon) if len(actual) >= 3 else MZResult(np.nan, np.nan, np.nan)
        else:
            mse_test = qlike_test = DMResult(np.nan, np.nan, np.nan)
            mz = MZResult(np.nan, np.nan, np.nan)
        benchmark_mse = float(np.mean(np.square(benchmark_frame["actual"] - benchmark_frame["forecast"])))
        benchmark_qlike = qlike(
            benchmark_frame["actual"], benchmark_frame["forecast"]
        )[1] if (benchmark_frame["actual"] > 0).all() and (benchmark_frame["forecast"] > 0).all() else np.nan
        records.append(
            {
                "model": name,
                "n": len(model_frame),
                "MSE": model_mse,
                "MAE": model_mae,
                "QLIKE": model_qlike,
                "MSE_ratio_to_benchmark": model_mse / benchmark_mse if benchmark_mse else np.nan,
                "QLIKE_ratio_to_benchmark": model_qlike / benchmark_qlike if benchmark_qlike else np.nan,
                "DM_MSE_p_raw": mse_test.p_value,
                "DM_MSE_p_holm": np.nan,
                "DM_QLIKE_p_raw": qlike_test.p_value,
                "DM_QLIKE_p_holm": np.nan,
                "MZ_a": mz.intercept,
                "MZ_b": mz.slope,
                "MZ_wald_p": mz.wald_p_value,
            }
        )
        index = len(records) - 1
        for field in ("DM_MSE_p_raw", "DM_QLIKE_p_raw"):
            value = records[index][field]
            if name != benchmark and isinstance(value, (int, float)) and np.isfinite(value):
                pvalue_locations.append((index, field.replace("_raw", "_holm")))
                pvalues.append(float(value))

    if pvalues:
        adjusted = multipletests(pvalues, method="holm")[1]
        for (row_index, field), value in zip(pvalue_locations, adjusted):
            records[row_index][field] = float(value)
    for record in records:
        if record["model"] == benchmark:
            record["DM_MSE_p_holm"] = record["DM_MSE_p_raw"]
            record["DM_QLIKE_p_holm"] = record["DM_QLIKE_p_raw"]

    summary = pd.DataFrame(records).sort_values("MSE", na_position="last").reset_index(drop=True)
    selected_names = summary.loc[summary["n"] > 0, "model"].head(top_n).tolist()
    summary.attrs["dm_pvalue_matrix_mse"] = dm_pvalue_matrix(
        results, models=selected_names, loss="mse", horizon=horizon
    )
    summary.attrs["dm_pvalue_matrix_qlike"] = dm_pvalue_matrix(
        results, models=selected_names, loss="qlike", horizon=horizon
    )
    return summary
