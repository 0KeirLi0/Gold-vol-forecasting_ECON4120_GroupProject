"""Common one-step forecasts and holdout comparison for daily volatility."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from statsmodels.tsa.api import VAR

from .auto_arima import fit_arima_order, hyndman_khandakar

MODEL_NAMES = (
    "yang_zhang_persistence",
    "expanding_mean",
    "ewma",
    "ar1",
    "ma1",
    "arma11",
    "auto_arima_hk",
    "arch1",
    "garch11",
    "gjr_garch",
    "egarch",
    "garch_t",
    "garchx_ohlc_range",
    "quantile_garch",
    "har",
    "harx_ohlc",
    "ridge_ohlc",
    "lasso_ohlc",
    "hist_gradient_boosting_ohlc",
    "var_ohlc",
)

MIN_LOG_VALUE = 1e-12
HAR_LAGS = 22
EXOGENOUS_COLUMNS = (
    "close_return",
    "overnight_log_return",
    "open_close_log_return",
    "high_low_log_range",
)


def _target(history: pd.DataFrame) -> np.ndarray:
    if "daily_volatility" not in history:
        raise ValueError("history must contain daily_volatility.")
    values = pd.to_numeric(history["daily_volatility"], errors="raise").to_numpy(
        dtype=float
    )
    if len(values) < 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("history must contain at least 3 finite non-negative targets.")
    return values


def _log_target(history: pd.DataFrame) -> np.ndarray:
    return np.log(np.maximum(_target(history), MIN_LOG_VALUE))


def _arima_forecast(history: pd.DataFrame, order: tuple[int, int, int]) -> float:
    values = _log_target(history)
    fit = fit_arima_order(values, order)
    prediction = float(np.asarray(fit.get_forecast(steps=1).predicted_mean)[0])
    return float(np.exp(prediction))


def _arch_forecast(
    history: pd.DataFrame,
    *,
    vol: str,
    p: int,
    o: int = 0,
    q: int = 0,
    distribution: str = "normal",
) -> float:
    values = _target(history) * 100
    if len(values) < 30:
        raise ValueError(f"{vol} requires at least 30 target observations.")
    fit = arch_model(
        values,
        mean="AR",
        lags=1,
        vol=vol,
        p=p,
        o=o,
        q=q,
        dist=distribution,
        rescale=False,
    ).fit(disp="off", show_warning=False)
    if int(getattr(fit, "convergence_flag", 1)) != 0:
        raise RuntimeError(f"{vol} fit did not converge.")
    return float(fit.forecast(horizon=1).mean.iloc[-1, 0]) / 100


def _lagged_design(history: pd.DataFrame, *, include_ohlc: bool, include_vix: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_vol = _log_target(history)
    if len(log_vol) <= HAR_LAGS + 3:
        raise ValueError(
            f"HAR models require more than {HAR_LAGS + 3} observations."
        )
    columns = list(EXOGENOUS_COLUMNS) + (["vix_close"] if include_vix else [])
    exogenous = (
        history.loc[:, columns].to_numpy(dtype=float) if include_ohlc else None
    )
    if exogenous is not None and not np.isfinite(exogenous).all():
        raise ValueError("HARX predictors must be finite through the forecast origin.")

    features: list[list[float]] = []
    targets: list[float] = []
    for target_position in range(HAR_LAGS, len(log_vol)):
        origin_position = target_position - 1
        row = [
            log_vol[origin_position],
            float(np.mean(log_vol[origin_position - 4 : origin_position + 1])),
            float(np.mean(log_vol[origin_position - HAR_LAGS + 1 : origin_position + 1])),
        ]
        if exogenous is not None:
            row.extend(exogenous[origin_position].tolist())
        features.append(row)
        targets.append(float(log_vol[target_position]))

    origin_position = len(log_vol) - 1
    forecast_features = [
        log_vol[origin_position],
        float(np.mean(log_vol[origin_position - 4 : origin_position + 1])),
        float(np.mean(log_vol[origin_position - HAR_LAGS + 1 : origin_position + 1])),
    ]
    if exogenous is not None:
        forecast_features.extend(exogenous[origin_position].tolist())
    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray([forecast_features], dtype=float),
    )


def _linear_forecast(
    history: pd.DataFrame, *, kind: str, include_ohlc: bool, include_vix: bool = False
) -> float:
    features, targets, forecast_features = _lagged_design(
        history, include_ohlc=include_ohlc, include_vix=include_vix
    )
    if kind == "har":
        design = np.column_stack((np.ones(len(features)), features))
        forecast_design = np.column_stack((np.ones(1), forecast_features))
        coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
        prediction = float((forecast_design @ coefficients).item())
    elif kind == "ridge":
        model = Ridge(alpha=1.0).fit(features, targets)
        prediction = float(model.predict(forecast_features)[0])
    elif kind == "lasso":
        model = Lasso(alpha=0.01, max_iter=50_000, tol=1e-5).fit(features, targets)
        prediction = float(model.predict(forecast_features)[0])
    elif kind == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=15, random_state=42
        ).fit(features, targets)
        prediction = float(model.predict(forecast_features)[0])
    else:
        raise ValueError(f"Unknown linear forecast kind: {kind}")
    return float(np.exp(prediction))


def _var_forecast(history: pd.DataFrame, *, include_vix: bool = False) -> float:
    values = _log_target(history)
    close_returns = pd.to_numeric(history["close_return"], errors="raise").to_numpy(
        dtype=float
    )
    components = [values, close_returns]
    if include_vix:
        components.append(
            pd.to_numeric(history["vix_close"], errors="raise").to_numpy(dtype=float)
        )
    observations = np.column_stack(components)
    if not np.isfinite(observations).all() or len(observations) < 10:
        raise ValueError("VAR requires 10 finite aligned volatility and return rows.")
    fit = VAR(observations).fit(maxlags=1, ic=None, trend="c")
    prediction = float(fit.forecast(observations[-fit.k_ar :], steps=1)[0, 0])
    return float(np.exp(prediction))


def _conditional_variance_forecast(
    history: pd.DataFrame, *, tail_threshold: float | None, include_vix: bool = False
) -> float:
    """Fit an AR(1) volatility mean with GARCH or threshold-GARCH errors."""
    values = _target(history)
    if len(values) < 100:
        raise ValueError("GARCH-X and quantile GARCH require 100 target observations.")
    exogenous = pd.to_numeric(
        history["high_low_log_range"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(exogenous).all() or (exogenous < 0).any():
        raise ValueError("GARCH-X requires finite non-negative OHLC range values.")
    range_scale = float(np.mean(exogenous))
    range_scaled = exogenous / max(range_scale, 1e-12)
    vix_scaled = None
    if include_vix:
        vix = pd.to_numeric(history["vix_close"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(vix).all() or (vix <= 0).any():
            raise ValueError("GARCH-X requires finite positive VIX values.")
        vix_scaled = vix / float(np.mean(vix))
    residual_scale = float(np.std(np.diff(values), ddof=1))
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("Conditional-variance model requires varying log volatility.")

    threshold = (
        float(np.quantile(np.abs(np.diff(values) / residual_scale), tail_threshold))
        if tail_threshold is not None
        else 0.0
    )
    initial_variance = max(float(np.var(values, ddof=1)), 1e-12)

    def paths(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        intercept, ar, omega, alpha, beta, extra = parameters[:6]
        residuals = np.zeros(len(values), dtype=float)
        residuals[1:] = (
            values[1:] - intercept - ar * values[:-1]
        ) / residual_scale
        variance = np.empty(len(values), dtype=float)
        variance[0] = initial_variance / residual_scale**2
        for index in range(1, len(values)):
            shock = residuals[index - 1] ** 2
            addition = (
                extra * range_scaled[index - 1]
                if tail_threshold is None
                else extra * shock * (abs(residuals[index - 1]) > threshold)
            )
            if vix_scaled is not None:
                addition += parameters[6] * vix_scaled[index - 1]
            variance[index] = (
                omega + alpha * shock + beta * variance[index - 1] + addition
            )
        return residuals, variance

    def objective(parameters: np.ndarray) -> float:
        residuals, variance = paths(parameters)
        if not np.isfinite(variance).all() or np.any(variance[1:] <= 0):
            return np.finfo(float).max
        losses = np.log(variance[1:]) + residuals[1:] ** 2 / variance[1:]
        score = 0.5 * float(np.sum(losses))
        return score if np.isfinite(score) else np.finfo(float).max

    variance_start = max(initial_variance / residual_scale**2 * 0.05, 1e-8)
    extra_start = 0.01
    starts = (
        [float(np.mean(values)), 0.1, variance_start, 0.05, 0.85, extra_start],
        [float(np.mean(values)), 0.5, variance_start * 2, 0.1, 0.7, extra_start],
    )
    if vix_scaled is not None:
        starts = tuple(start + [0.01] for start in starts)
    extra_upper = 100.0
    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: (
                0.999 - x[3] - x[4]
                if tail_threshold is None
                else 0.999 - x[3] - x[4] - x[5]
            ),
        }
    ]
    fits = [
        minimize(
            objective,
            np.asarray(start),
            method="SLSQP",
            bounds=[
                (None, None),
                (-0.999, 0.999),
                (1e-8, 100.0),
                (0.0, 0.999),
                (0.0, 0.999),
                (0.0, extra_upper),
            ] + ([(0.0, extra_upper)] if vix_scaled is not None else []),
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        for start in starts
    ]
    valid = [
        fit
        for fit in fits
        if fit.success
        and np.isfinite(fit.fun)
        and np.isfinite(fit.x).all()
        and fit.x[2] > 0
        and fit.x[3] >= 0
        and fit.x[4] >= 0
        and (vix_scaled is None or fit.x[6] >= 0)
        and fit.x[3] + fit.x[4] + (
            fit.x[5] if tail_threshold is not None else 0.0
        ) <= 0.999 + 1e-8
    ]
    if not valid:
        messages = "; ".join(str(fit.message) for fit in fits)
        raise RuntimeError(f"Conditional-variance fit failed: {messages}")
    fit = min(valid, key=lambda item: item.fun)
    prediction = float(fit.x[0] + fit.x[1] * values[-1])
    return prediction


def forecast_daily_volatility(
    model: str,
    history: pd.DataFrame,
    *,
    auto_arima_order: tuple[int, int, int],
    include_vix: bool = False,
) -> float:
    """Return one positive forecast of daily volatility from history through origin."""
    values = _target(history)
    if model == "yang_zhang_persistence":
        prediction = float(values[-1])
    elif model == "expanding_mean":
        prediction = float(np.mean(values))
    elif model == "ewma":
        estimate = float(values[0])
        for observation in values[1:]:
            estimate = 0.94 * estimate + 0.06 * float(observation)
        prediction = estimate
    elif model == "ar1":
        prediction = _arima_forecast(history, (1, 0, 0))
    elif model == "ma1":
        prediction = _arima_forecast(history, (0, 0, 1))
    elif model == "arma11":
        prediction = _arima_forecast(history, (1, 0, 1))
    elif model == "auto_arima_hk":
        prediction = _arima_forecast(history, auto_arima_order)
    elif model == "arch1":
        prediction = _arch_forecast(history, vol="ARCH", p=1)
    elif model == "garch11":
        prediction = _arch_forecast(history, vol="GARCH", p=1, q=1)
    elif model == "gjr_garch":
        prediction = _arch_forecast(history, vol="GARCH", p=1, o=1, q=1)
    elif model == "egarch":
        prediction = _arch_forecast(history, vol="EGARCH", p=1, o=1, q=1)
    elif model == "garch_t":
        prediction = _arch_forecast(
            history, vol="GARCH", p=1, q=1, distribution="t"
        )
    elif model == "garchx_ohlc_range":
        prediction = _conditional_variance_forecast(history, tail_threshold=None, include_vix=include_vix)
    elif model == "quantile_garch":
        prediction = _conditional_variance_forecast(history, tail_threshold=0.95)
    elif model == "har":
        prediction = _linear_forecast(history, kind="har", include_ohlc=False)
    elif model == "harx_ohlc":
        prediction = _linear_forecast(history, kind="har", include_ohlc=True, include_vix=include_vix)
    elif model == "ridge_ohlc":
        prediction = _linear_forecast(history, kind="ridge", include_ohlc=True, include_vix=include_vix)
    elif model == "lasso_ohlc":
        prediction = _linear_forecast(history, kind="lasso", include_ohlc=True, include_vix=include_vix)
    elif model == "hist_gradient_boosting_ohlc":
        prediction = _linear_forecast(
            history, kind="hist_gradient_boosting", include_ohlc=True, include_vix=include_vix
        )
    elif model == "var_ohlc":
        prediction = _var_forecast(history, include_vix=include_vix)
    else:
        raise ValueError(f"Unknown daily-volatility model: {model}")

    if not np.isfinite(prediction) or prediction <= 0:
        raise RuntimeError(f"{model} returned a non-positive or non-finite forecast.")
    return float(prediction)


def _metrics(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, float, float]:
    errors = forecast - actual
    return (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.mean(errors)),
    )


def _forecast_origin(
    task: tuple[
        pd.DataFrame,
        pd.Timestamp,
        pd.Timestamp,
        float,
        tuple[int, int, int],
        tuple[str, ...],
        bool,
    ],
) -> list[dict[str, object]]:
    history, origin, target_date, actual, order, model_names, include_vix = task
    rows: list[dict[str, object]] = []
    for model in model_names:
        try:
            forecast = forecast_daily_volatility(
                model, history, auto_arima_order=order, include_vix=include_vix
            )
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
                "model": model,
                "forecast": forecast,
                "actual": actual,
                "converged": not error,
                "error": error,
                "history_count": len(history),
            }
        )
    return rows


def evaluate_daily_volatility(
    data: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    model_names: Sequence[str] = MODEL_NAMES,
    workers: int = 1,
    progress: bool = True,
    include_vix: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Evaluate models on a chronological split with sequential one-step updates.

    Parameters are refit at each test origin using the initial training sample
    plus actual test targets observed before that origin. The test target itself
    is never included in its own forecast history.
    """
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one.")
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("model_names must contain unique model names.")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    unknown = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown daily-volatility models: {unknown}")
    required = {
        "date",
        "daily_volatility",
        "close_return",
        "overnight_log_return",
        "open_close_log_return",
        "high_low_log_range",
    }
    if include_vix:
        required.add("vix_close")
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("data dates must be nonmissing and unique.")
    frame = frame.sort_values("date").dropna(subset=["daily_volatility"]).reset_index(
        drop=True
    )
    if len(frame) < 60:
        raise ValueError("At least 60 valid daily-volatility observations are required.")
    targets = pd.to_numeric(frame["daily_volatility"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("daily_volatility must contain finite non-negative values.")
    if include_vix:
        vix = pd.to_numeric(frame["vix_close"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(vix).all() or (vix <= 0).any():
            raise ValueError("VIX must contain finite positive values on all target dates.")
    split_position = int(np.floor(len(frame) * train_fraction))
    if split_position < 30 or split_position >= len(frame):
        raise ValueError("The chronological split leaves too few training or test rows.")

    fixed_order: tuple[int, int, int] | None = None
    if "auto_arima_hk" in model_names:
        selection = hyndman_khandakar(
            np.log(np.maximum(targets[:split_position], MIN_LOG_VALUE)),
            max_p=3,
            max_q=3,
            max_d=2,
            max_order=4,
        )
        fixed_order = selection.order

    attempts: list[dict[str, object]] = []
    test_count = len(frame) - split_position
    order = fixed_order or (1, 0, 0)

    def tasks():
        for test_position in range(split_position, len(frame)):
            history = frame.iloc[:test_position].copy()
            yield (
                history,
                pd.Timestamp(history.iloc[-1]["date"]),
                pd.Timestamp(frame.at[test_position, "date"]),
                float(frame.at[test_position, "daily_volatility"]),
                order,
                tuple(model_names),
                include_vix,
            )

    if workers == 1:
        completed_origins = map(_forecast_origin, tasks())
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        completed_origins = executor.map(_forecast_origin, tasks(), chunksize=1)
    try:
        for completed, origin_rows in enumerate(completed_origins, start=1):
            attempts.extend(origin_rows)
            if progress and (completed % 25 == 0 or completed == test_count):
                print(f"Completed {completed}/{test_count} test dates.", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    results = pd.DataFrame(attempts)
    successful = results.loc[
        results["converged"] & np.isfinite(results["forecast"])
    ]
    common_dates = set(frame.iloc[split_position:]["date"])
    for model in model_names:
        valid_dates = set(
            successful.loc[successful["model"].eq(model), "target_date"]
        )
        common_dates.intersection_update(valid_dates)
    common = successful.loc[successful["target_date"].isin(common_dates)]

    rows: list[dict[str, object]] = []
    for model in model_names:
        own = successful.loc[successful["model"].eq(model)]
        own_mae, own_rmse, own_bias = (
            _metrics(
                own["actual"].to_numpy(dtype=float),
                own["forecast"].to_numpy(dtype=float),
            )
            if not own.empty
            else (np.nan, np.nan, np.nan)
        )
        shared = common.loc[common["model"].eq(model)]
        common_mae, common_rmse, common_bias = (
            _metrics(
                shared["actual"].to_numpy(dtype=float),
                shared["forecast"].to_numpy(dtype=float),
            )
            if not shared.empty
            else (np.nan, np.nan, np.nan)
        )
        rows.append(
            {
                "model": model,
                "n_successful": len(own),
                "failure_count": int((~results.loc[results["model"].eq(model), "converged"]).sum()),
                "MAE": own_mae,
                "RMSE": own_rmse,
                "bias": own_bias,
                "common_n": len(shared),
                "common_MAE": common_mae,
                "common_RMSE": common_rmse,
                "common_bias": common_bias,
            }
        )
    summary = pd.DataFrame(rows)
    summary["rank_common_MAE"] = summary["common_MAE"].rank(
        method="min", na_option="bottom"
    ).astype("Int64")
    summary["rank_common_RMSE"] = summary["common_RMSE"].rank(
        method="min", na_option="bottom"
    ).astype("Int64")
    summary = summary.sort_values(
        ["rank_common_MAE", "rank_common_RMSE", "model"], ignore_index=True
    )

    manifest: dict[str, object] = {
        "target": "daily_volatility",
        "include_vix": include_vix,
        "estimator": "Yang-Zhang",
        "target_windows_overlap": True,
        "train_fraction": train_fraction,
        "test_fraction": 1 - train_fraction,
        "split_method": "chronological by valid target observations",
        "test_forecast_method": (
            "sequential one-step; refit on initial training data and prior "
            "observed test targets at each origin"
        ),
        "model_names": list(model_names),
        "fixed_auto_arima_order": list(fixed_order) if fixed_order else None,
        "workers": workers,
        "training_observations": split_position,
        "test_observations": test_count,
        "common_test_observations": len(common_dates),
        "test_start": pd.Timestamp(frame.iloc[split_position]["date"]).date().isoformat(),
        "test_end": pd.Timestamp(frame.iloc[-1]["date"]).date().isoformat(),
        "ranking_metrics": ["MAE", "RMSE"],
        "statistical_significance_tests": False,
        "failures": {
            str(model): int(
                (~results.loc[results["model"].eq(model), "converged"]).sum()
            )
            for model in model_names
        },
    }
    return results, summary, manifest


def save_daily_volatility_evaluation(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: dict[str, object],
    output_dir: str | Path,
    *,
    input_path: str | Path,
) -> None:
    """Save forecasts, metrics, and reproducibility manifest to processed data."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / "stage4_yz_forecasts.csv", index=False)
    summary.to_csv(directory / "stage4_yz_summary.csv", index=False)
    output_manifest = dict(manifest)
    output_manifest["input_path"] = str(Path(input_path))
    output_manifest["input_sha256"] = hashlib.sha256(
        Path(input_path).read_bytes()
    ).hexdigest()
    (directory / "stage4_yz_manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
