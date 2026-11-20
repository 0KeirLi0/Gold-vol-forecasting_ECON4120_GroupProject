"""Compare log-target, bias-corrected Yang-Zhang forecasts for Samuel."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import t as student_t
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = (
    ROOT / "data/processed/stage2_yz_selected/GCc1_yang_zhang_volatility.csv"
)
BASE_FORECASTS = ROOT / "data/processed/stage2_yz_forecasts.csv"
SELECTED_FORECASTS = (
    ROOT / "data/processed/stage2_yz_selected/stage2_yz_forecasts.csv"
)
EGARCHX_FORECASTS = (
    ROOT / "outputs/samuel/stage3_q5/egarchx_regenerated_forecasts.csv"
)
OUTPUT_DIR = ROOT / "outputs/samuel/stage3_q5/log_transform"

MODEL_IDS = (
    "garch11",
    "egarch",
    "garchx_ohlc_range",
    "egarchx",
    "harx_ohlc",
    "hist_gradient_boosting_ohlc",
)
MODEL_LABELS = {
    "garch11": "GARCH(1,1)",
    "egarch": "EGARCH(1,1)",
    "garchx_ohlc_range": "Range GARCH-X",
    "egarchx": "EGARCH-X",
    "harx_ohlc": "HAR-X (OHLC)",
    "hist_gradient_boosting_ohlc": "HGB (OHLC)",
}
BASE_MODEL_SOURCES = {
    "garch11": BASE_FORECASTS,
    "egarch": BASE_FORECASTS,
    "garchx_ohlc_range": SELECTED_FORECASTS,
    "harx_ohlc": SELECTED_FORECASTS,
    "hist_gradient_boosting_ohlc": SELECTED_FORECASTS,
    "egarchx": EGARCHX_FORECASTS,
}
EXOGENOUS_COLUMNS = (
    "close_return",
    "overnight_log_return",
    "open_close_log_return",
    "high_low_log_range",
)
SELECTED_PREDICTORS = ("dxy_close", "gold_iv_1m", "gold_iv_1y")
HAR_LAGS = 22
QUINTILE_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")
NW_BANDWIDTHS = (1, 4)
RANDOM_SEED = 42


def _positive_finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{label} must contain finite, strictly positive values.")


def _lognormal_mean(log_location: float, log_variance: float) -> float:
    if not np.isfinite(log_location) or not np.isfinite(log_variance):
        raise ValueError("Log location and variance must be finite.")
    if log_variance <= 0:
        raise ValueError("Log-scale residual variance must be positive.")
    exponent = log_location + 0.5 * log_variance
    if not -700 < exponent < 700:
        raise RuntimeError("Bias-corrected log forecast is outside the finite range.")
    return float(np.exp(exponent))


def _arch_log_forecast(
    log_target: np.ndarray, *, volatility: str
) -> tuple[float, float, str]:
    options: dict[str, int | str] = {
        "mean": "AR",
        "lags": 1,
        "vol": volatility,
        "p": 1,
        "o": 1 if volatility == "EGARCH" else 0,
        "q": 1,
        "rescale": False,
    }
    fit = arch_model(log_target, **options).fit(disp="off", show_warning=False)
    if int(getattr(fit, "convergence_flag", 1)) != 0:
        raise RuntimeError(f"{volatility} log-target fit did not converge.")
    forecast = fit.forecast(horizon=1)
    location = float(forecast.mean.iloc[-1, 0])
    variance = float(forecast.variance.iloc[-1, 0])
    return location, variance, "one-step conditional variance"


def _garchx_log_forecast(history: pd.DataFrame, log_target: np.ndarray) -> tuple[float, float, str]:
    """Fit the existing AR(1)-mean, range-in-variance Gaussian QL model."""
    ranges = pd.to_numeric(history["high_low_log_range"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(ranges).all() or (ranges < 0).any():
        raise ValueError("GARCH-X range inputs must be finite and non-negative.")
    range_scale = float(np.mean(ranges))
    range_scaled = ranges / max(range_scale, 1e-12)
    residual_scale = float(np.std(np.diff(log_target), ddof=1))
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("GARCH-X log target must vary.")
    initial_variance = max(float(np.var(log_target, ddof=1)), 1e-12)
    initial = initial_variance / residual_scale**2

    def paths(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        intercept, ar, omega, alpha, beta, gamma = parameters
        residuals = np.zeros(len(log_target), dtype=float)
        residuals[1:] = (
            log_target[1:] - intercept - ar * log_target[:-1]
        ) / residual_scale
        variance = np.empty(len(log_target), dtype=float)
        variance[0] = initial
        for position in range(1, len(log_target)):
            variance[position] = (
                omega
                + alpha * residuals[position - 1] ** 2
                + gamma * range_scaled[position - 1]
                + beta * variance[position - 1]
            )
        return residuals, variance

    def objective(parameters: np.ndarray) -> float:
        residuals, variance = paths(parameters)
        if not np.isfinite(variance).all() or np.any(variance[1:] <= 0):
            return np.finfo(float).max
        score = 0.5 * float(
            np.sum(np.log(variance[1:]) + residuals[1:] ** 2 / variance[1:])
        )
        return score if np.isfinite(score) else np.finfo(float).max

    variance_start = max(initial * 0.05, 1e-8)
    starts = (
        np.array([float(np.mean(log_target)), 0.1, variance_start, 0.05, 0.85, 0.01]),
        np.array([float(np.mean(log_target)), 0.5, 2 * variance_start, 0.1, 0.7, 0.01]),
    )
    constraints = [
        {"type": "ineq", "fun": lambda values: 0.999 - values[3] - values[4]}
    ]
    fits = [
        minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[
                (None, None),
                (-0.999, 0.999),
                (1e-8, 100.0),
                (0.0, 0.999),
                (0.0, 0.999),
                (0.0, 100.0),
            ],
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
        and fit.x[3] + fit.x[4] <= 0.999 + 1e-8
    ]
    if not valid:
        messages = "; ".join(str(fit.message) for fit in fits)
        raise RuntimeError(f"GARCH-X log-target fit failed: {messages}")
    fit = min(valid, key=lambda candidate: candidate.fun)
    intercept, ar, omega, alpha, beta, gamma = fit.x
    residuals, variance_path = paths(fit.x)
    next_variance_scaled = (
        omega
        + alpha * residuals[-1] ** 2
        + gamma * range_scaled[-1]
        + beta * variance_path[-1]
    )
    log_variance = float(next_variance_scaled * residual_scale**2)
    location = float(intercept + ar * log_target[-1])
    return location, log_variance, "one-step conditional variance"


def _linear_features(
    history: pd.DataFrame, log_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = (*EXOGENOUS_COLUMNS, *SELECTED_PREDICTORS)
    exogenous = history.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(exogenous).all():
        raise ValueError("HAR-X predictors must be finite through the origin.")
    if len(log_target) <= HAR_LAGS + 3:
        raise ValueError("HAR-X and HGB require more than 25 history observations.")

    features: list[list[float]] = []
    targets: list[float] = []
    for target_position in range(HAR_LAGS, len(log_target)):
        origin_position = target_position - 1
        features.append(
            [
                log_target[origin_position],
                float(np.mean(log_target[origin_position - 4 : origin_position + 1])),
                float(
                    np.mean(
                        log_target[
                            origin_position - HAR_LAGS + 1 : origin_position + 1
                        ]
                    )
                ),
                *exogenous[origin_position].tolist(),
            ]
        )
        targets.append(float(log_target[target_position]))
    origin_position = len(log_target) - 1
    forecast_features = np.array(
        [
            log_target[origin_position],
            float(np.mean(log_target[origin_position - 4 : origin_position + 1])),
            float(
                np.mean(
                    log_target[
                        origin_position - HAR_LAGS + 1 : origin_position + 1
                    ]
                )
            ),
            *exogenous[origin_position].tolist(),
        ],
        dtype=float,
    )
    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
        forecast_features.reshape(1, -1),
    )


def _harx_log_forecast(history: pd.DataFrame, log_target: np.ndarray) -> tuple[float, float, str]:
    features, targets, forecast_features = _linear_features(history, log_target)
    design = np.column_stack((np.ones(len(features)), features))
    forecast_design = np.column_stack((np.ones(1), forecast_features))
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    residuals = targets - design @ coefficients
    degrees_freedom = len(targets) - design.shape[1]
    if degrees_freedom <= 0:
        raise ValueError("HAR-X has no residual degrees of freedom.")
    variance = float(np.sum(np.square(residuals)) / degrees_freedom)
    location = float((forecast_design @ coefficients).item())
    return location, variance, "training OLS residual variance"


def _hgb_log_forecast(history: pd.DataFrame, log_target: np.ndarray) -> tuple[float, float, str]:
    features, targets, forecast_features = _linear_features(history, log_target)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        random_state=RANDOM_SEED,
    ).fit(features, targets)
    residuals = targets - model.predict(features)
    variance = float(np.mean(np.square(residuals)))
    location = float(model.predict(forecast_features)[0])
    return location, variance, "training in-sample HGB residual variance"


def _fit_log_model(
    history: pd.DataFrame, model: str
) -> tuple[float, float, str]:
    values = pd.to_numeric(history["daily_volatility"], errors="raise").to_numpy(
        dtype=float
    )
    _positive_finite(values, "Training Yang-Zhang volatility")
    log_target = np.log(values)
    if model == "garch11":
        return _arch_log_forecast(log_target, volatility="GARCH")
    if model == "egarch":
        return _arch_log_forecast(log_target, volatility="EGARCH")
    if model == "garchx_ohlc_range":
        return _garchx_log_forecast(history, log_target)
    if model == "egarchx":
        predictors = history.loc[:, SELECTED_PREDICTORS].to_numpy(dtype=float)
        if not np.isfinite(predictors).all() or (predictors <= 0).any():
            raise ValueError("EGARCH-X predictors must be finite and positive.")
        exogenous = predictors / predictors.mean(axis=0)
        lagged_exogenous = exogenous.copy()
        lagged_exogenous[1:] = exogenous[:-1]
        fit = arch_model(
            log_target,
            mean="ARX",
            lags=1,
            x=lagged_exogenous,
            vol="EGARCH",
            p=1,
            o=1,
            q=1,
            rescale=False,
        ).fit(disp="off", show_warning=False)
        if int(getattr(fit, "convergence_flag", 1)) != 0:
            raise RuntimeError("EGARCH-X log-target fit did not converge.")
        forecast_x = exogenous[-1].reshape(len(SELECTED_PREDICTORS), 1, 1)
        forecast = fit.forecast(horizon=1, x=forecast_x)
        location = float(forecast.mean.iloc[-1, 0])
        variance = float(forecast.variance.iloc[-1, 0])
        return location, variance, "one-step conditional variance"
    if model == "harx_ohlc":
        return _harx_log_forecast(history, log_target)
    if model == "hist_gradient_boosting_ohlc":
        return _hgb_log_forecast(history, log_target)
    raise ValueError(f"Unknown model: {model}")


def _forecast_origin(
    task: tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, float]
) -> list[dict[str, object]]:
    history, origin_date, target_date, actual = task
    rows: list[dict[str, object]] = []
    for model in MODEL_IDS:
        try:
            location, variance, variance_method = _fit_log_model(history, model)
            forecast = _lognormal_mean(location, variance)
            rows.append(
                {
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "actual": actual,
                    "log_location": location,
                    "log_variance": variance,
                    "log_sigma": float(np.sqrt(variance)),
                    "bias_correction_factor": float(np.exp(0.5 * variance)),
                    "forecast": forecast,
                    "variance_method": variance_method,
                    "converged": True,
                    "error": "",
                }
            )
        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ) as exception:
            rows.append(
                {
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "actual": actual,
                    "log_location": np.nan,
                    "log_variance": np.nan,
                    "log_sigma": np.nan,
                    "bias_correction_factor": np.nan,
                    "forecast": np.nan,
                    "variance_method": "",
                    "converged": False,
                    "error": f"{type(exception).__name__}: {exception}",
                }
            )
    return rows


def _baseline_forecasts(test_dates: pd.DatetimeIndex) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODEL_IDS:
        path = BASE_MODEL_SOURCES[model]
        frame = pd.read_csv(path)
        selected = frame.loc[frame["model"].eq(model), ["target_date", "forecast", "actual"]].copy()
        if selected.empty:
            raise ValueError(f"{path} does not contain model {model!r}.")
        selected["target_date"] = pd.to_datetime(selected["target_date"], errors="raise")
        selected = selected.loc[selected["target_date"].isin(test_dates)].copy()
        selected["forecast"] = pd.to_numeric(selected["forecast"], errors="raise")
        selected["actual"] = pd.to_numeric(selected["actual"], errors="raise")
        selected = selected.rename(columns={"forecast": "baseline_forecast"})
        selected["model"] = model
        selected["baseline_source"] = str(path.relative_to(ROOT))
        frames.append(selected)
    baseline = pd.concat(frames, ignore_index=True)
    if baseline.duplicated(["target_date", "model"]).any():
        raise ValueError("Baseline forecasts contain duplicate model/date rows.")
    expected = pd.MultiIndex.from_product(
        [test_dates, MODEL_IDS], names=["target_date", "model"]
    )
    actual = pd.MultiIndex.from_frame(baseline[["target_date", "model"]])
    if set(actual) != set(expected):
        raise ValueError("Baseline model forecasts do not cover the common test dates.")
    if not np.isfinite(
        baseline[["baseline_forecast", "actual"]].to_numpy(dtype=float)
    ).all() or (baseline["baseline_forecast"] <= 0).any():
        raise ValueError("Baseline forecasts must be finite and positive.")
    return baseline


def _dm_test(loss_difference: np.ndarray, bandwidth: int) -> tuple[float, float]:
    count = len(loss_difference)
    mean_difference = float(np.mean(loss_difference))
    centered = loss_difference - mean_difference
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, min(bandwidth, count - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2 * (1 - lag / (bandwidth + 1)) * covariance
    long_run_variance = max(long_run_variance, 0.0)
    if long_run_variance == 0:
        statistic = 0.0 if mean_difference == 0 else np.sign(mean_difference) * np.inf
    else:
        statistic = mean_difference / np.sqrt(long_run_variance / count)
    horizon = 1
    hln_squared = (
        count + 1 - 2 * horizon + horizon * (horizon - 1) / count
    ) / count
    corrected = statistic * np.sqrt(max(hln_squared, 0.0))
    p_value = (
        float(2 * student_t.sf(abs(corrected), df=count - 1))
        if np.isfinite(corrected)
        else 0.0
    )
    return float(corrected), p_value


def _comparison_tables(
    forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    successful = forecasts.loc[forecasts["converged"]].copy()
    successful["error"] = successful["actual"] - successful["forecast"]
    successful["baseline_error"] = (
        successful["actual"] - successful["baseline_forecast"]
    )
    metrics: list[dict[str, object]] = []
    grouped: list[dict[str, object]] = []
    for model in MODEL_IDS:
        for variant, forecast_column, error_column in (
            ("baseline", "baseline_forecast", "baseline_error"),
            ("log_target_bias_corrected", "forecast", "error"),
        ):
            frame = successful.loc[successful["model"].eq(model)].copy()
            if frame.empty:
                continue
            errors = frame[error_column].to_numpy(dtype=float)
            metrics.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "variant": variant,
                    "n": len(frame),
                    "MAE": float(np.mean(np.abs(errors))),
                    "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
                    "mean_error_actual_minus_forecast": float(np.mean(errors)),
                    "residual_variance": float(np.var(errors, ddof=1)),
                    "first_target": frame["target_date"].min().date().isoformat(),
                    "last_target": frame["target_date"].max().date().isoformat(),
                    "mean_log_sigma": (
                        float(frame["log_sigma"].mean())
                        if variant == "log_target_bias_corrected"
                        else np.nan
                    ),
                    "mean_bias_correction_factor": (
                        float(frame["bias_correction_factor"].mean())
                        if variant == "log_target_bias_corrected"
                        else np.nan
                    ),
                }
            )
            forecast_column = (
                "baseline_forecast" if variant == "baseline" else "forecast"
            )
            forecast_order = frame.sort_values(
                [forecast_column, "target_date"], kind="mergesort"
            )
            forecast_order["forecast_quintile"] = pd.qcut(
                np.arange(len(forecast_order)),
                q=5,
                labels=QUINTILE_LABELS,
            ).astype(str)
            for quintile, group in forecast_order.groupby(
                "forecast_quintile", sort=True
            ):
                group_errors = group[error_column].to_numpy(dtype=float)
                grouped.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "variant": variant,
                        "forecast_quintile": str(quintile),
                        "n": len(group),
                        "mean_error": float(np.mean(group_errors)),
                        "residual_variance": float(np.var(group_errors, ddof=1)),
                        "MAE": float(np.mean(np.abs(group_errors))),
                        "RMSE": float(np.sqrt(np.mean(np.square(group_errors)))),
                    }
                )

    metrics_frame = pd.DataFrame(metrics)
    grouped_frame = pd.DataFrame(grouped)
    q5_rows: list[dict[str, object]] = []
    for model in MODEL_IDS:
        model_frame = successful.loc[successful["model"].eq(model)].copy()
        ordered = model_frame.sort_values(
            ["baseline_forecast", "target_date"], kind="mergesort"
        ).reset_index(drop=True)
        ordered["baseline_quintile"] = pd.qcut(
            np.arange(len(ordered)), q=5, labels=QUINTILE_LABELS
        ).astype(str)
        ordered["baseline_q5"] = ordered["baseline_quintile"].eq("Q5")
        base_q5 = ordered.loc[ordered["baseline_q5"]]
        base_q5_variance = float(
            np.var(base_q5["baseline_error"].to_numpy(dtype=float), ddof=1)
        )
        transform_q5 = model_frame.sort_values(
            ["forecast", "target_date"], kind="mergesort"
        ).reset_index(drop=True)
        transform_q5["forecast_quintile"] = pd.qcut(
            np.arange(len(transform_q5)), q=5, labels=QUINTILE_LABELS
        ).astype(str)
        transform_q5 = transform_q5.loc[
            transform_q5["forecast_quintile"].eq("Q5")
        ]
        own_q5_variance = float(
            np.var(transform_q5["error"].to_numpy(dtype=float), ddof=1)
        )
        same_dates_transformed = base_q5["error"].to_numpy(dtype=float)
        same_dates_base = base_q5["baseline_error"].to_numpy(dtype=float)
        same_dates_transformed_variance = float(
            np.var(same_dates_transformed, ddof=1)
        )
        q5_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "baseline_q5_n": len(base_q5),
                "baseline_q5_variance": base_q5_variance,
                "transformed_error_variance_on_baseline_q5_dates": (
                    same_dates_transformed_variance
                ),
                "same_dates_variance_reduction_pct": (
                    100
                    * (base_q5_variance - same_dates_transformed_variance)
                    / base_q5_variance
                ),
                "transformed_own_q5_n": len(transform_q5),
                "transformed_own_q5_variance": own_q5_variance,
            }
        )

    dm_rows: list[dict[str, object]] = []
    for model in MODEL_IDS:
        frame = successful.loc[successful["model"].eq(model)].sort_values(
            "target_date"
        )
        for loss in ("absolute_error", "squared_error"):
            transformed_loss = (
                np.abs(frame["error"].to_numpy(dtype=float))
                if loss == "absolute_error"
                else np.square(frame["error"].to_numpy(dtype=float))
            )
            baseline_loss = (
                np.abs(frame["baseline_error"].to_numpy(dtype=float))
                if loss == "absolute_error"
                else np.square(frame["baseline_error"].to_numpy(dtype=float))
            )
            difference = transformed_loss - baseline_loss
            for bandwidth in NW_BANDWIDTHS:
                statistic, p_value = _dm_test(difference, bandwidth)
                dm_rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "loss": loss,
                        "bandwidth": bandwidth,
                        "n": len(frame),
                        "mean_loss_difference_transformed_minus_baseline": float(
                            np.mean(difference)
                        ),
                        "DM_HLN_statistic": statistic,
                        "p_raw": p_value,
                    }
                )
    dm_frame = pd.DataFrame(dm_rows)
    dm_frame["p_holm"] = np.nan
    for _, indices in dm_frame.groupby(["loss", "bandwidth"]).groups.items():
        dm_frame.loc[indices, "p_holm"] = multipletests(
            dm_frame.loc[indices, "p_raw"].to_numpy(dtype=float), method="holm"
        )[1]
    return metrics_frame, grouped_frame, pd.DataFrame(q5_rows), dm_frame


def _write_report(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    q5: pd.DataFrame,
    dm: pd.DataFrame,
) -> None:
    attempt_summary = (
        forecasts.groupby("model_label", sort=False)["converged"]
        .agg(attempts="size", failures=lambda values: int((~values).sum()))
        .reset_index()
    )
    transform_metrics = metrics.loc[
        metrics["variant"].eq("log_target_bias_corrected")
    ].set_index("model")
    baseline_metrics = metrics.loc[metrics["variant"].eq("baseline")].set_index(
        "model"
    )
    comparison = pd.DataFrame(
        {
            "model_label": [
                MODEL_LABELS[model] for model in MODEL_IDS
            ],
            "baseline_MAE": [
                baseline_metrics.loc[model, "MAE"] for model in MODEL_IDS
            ],
            "logY_corrected_MAE": [
                transform_metrics.loc[model, "MAE"] for model in MODEL_IDS
            ],
            "baseline_RMSE": [
                baseline_metrics.loc[model, "RMSE"] for model in MODEL_IDS
            ],
            "logY_corrected_RMSE": [
                transform_metrics.loc[model, "RMSE"] for model in MODEL_IDS
            ],
            "baseline_mean_error": [
                baseline_metrics.loc[model, "mean_error_actual_minus_forecast"]
                for model in MODEL_IDS
            ],
            "logY_corrected_mean_error": [
                transform_metrics.loc[
                    model, "mean_error_actual_minus_forecast"
                ]
                for model in MODEL_IDS
            ],
        }
    )
    lines = [
        "# Log-target and bias-corrected forecast experiment",
        "",
        "All six model specifications were fit to `log(daily_volatility)` using only observations available through each forecast origin. Forecasts are back-transformed as `exp(mu_hat + 0.5 * sigma_hat^2)`, the conditional-mean correction under a normal log-error assumption. It differs from `exp(mu_hat)`, which estimates the conditional median.",
        "",
        "GARCH and EGARCH use their one-step conditional variance on the log target. Range GARCH-X keeps its range predictor in the conditional-variance equation and uses Gaussian quasi-likelihood. EGARCH-X keeps DXY and gold IV predictors in the ARX mean. HAR-X uses the existing OHLC and selected predictors in its log-scale OLS mean. HGB uses its existing log-scale features and fixed random seed 42. For HAR-X/HGB, `sigma_hat^2` is estimated from training residuals; HGB uses in-sample training residual variance, which may be optimistic.",
        "",
        "## Sample and convergence",
        "",
        f"- Test targets: {forecasts['target_date'].nunique()} ({forecasts['target_date'].min().date()} through {forecasts['target_date'].max().date()}).",
        "- Forecast errors are actual minus forecast; positive mean error indicates under-forecasting.",
        "- Each transformed model is compared with its existing saved forecast on the same target dates.",
        "",
        attempt_summary.to_markdown(index=False),
        "",
        "## Overall performance",
        "",
        comparison.to_markdown(index=False, floatfmt=".7g"),
        "",
        "## Forecast-Q5 variance",
        "",
        "The first two variance columns compare base and transformed errors on the exact same dates classified as baseline forecast Q5. The final column is also shown, but uses the transformed model's own holdout forecast ranking, so it can contain different dates.",
        "",
        q5.to_markdown(index=False, floatfmt=".7g"),
        "",
        "## Paired tests",
        "",
        "Diebold-Mariano tests compare transformed minus baseline loss, use Newey-West bandwidths 1 and 4, the Harvey-Leybourne-Newbold small-sample correction with one-step horizon, and Holm adjustment across six models within each loss/bandwidth family. Negative mean loss differences favor the transformed forecast.",
        "",
        dm.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Interpretation limits",
        "",
        "- The correction assumes conditionally normal log errors; HGB and HAR-X residuals need not be normal.",
        "- The variance estimates use only information through the origin, but HAR-X/HGB use in-sample training residuals.",
        "- Q5 membership is based on holdout forecast ranks and is retrospective; same-date Q5 variances are included to separate membership changes from forecast changes.",
        "- This is a single holdout experiment, not evidence of causal effects or durable model superiority.",
        "",
    ]
    (OUTPUT_DIR / "log_transform_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    np.random.seed(RANDOM_SEED)
    data = pd.read_csv(TARGET_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.dropna(subset=["daily_volatility"]).sort_values("date").reset_index(
        drop=True
    )
    targets = pd.to_numeric(data["daily_volatility"], errors="raise").to_numpy(
        dtype=float
    )
    _positive_finite(targets, "Yang-Zhang volatility")
    if data["date"].duplicated().any():
        raise ValueError("Prepared targets contain duplicate dates.")
    if not data["split"].isin(("train", "test")).all():
        raise ValueError("Prepared targets contain unknown split labels.")
    test_positions = np.flatnonzero(data["split"].eq("test").to_numpy())
    if not len(test_positions) or not data["split"].eq("train").iloc[: test_positions[0]].all():
        raise ValueError("Prepared data does not have the expected chronological split.")
    if not data["split"].eq("test").iloc[test_positions[0] :].all():
        raise ValueError("Prepared split is not chronological.")
    test_dates = pd.DatetimeIndex(data.loc[test_positions, "date"])
    baseline = _baseline_forecasts(test_dates)
    baseline_actuals = baseline.drop_duplicates("target_date").set_index("target_date")[
        "actual"
    ].sort_index()
    target_actuals = data.set_index("date").loc[test_dates, "daily_volatility"]
    if not np.allclose(
        baseline_actuals.to_numpy(dtype=float),
        target_actuals.to_numpy(dtype=float),
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("Saved forecasts and transformed target data have different actuals.")

    def tasks():
        for position in test_positions:
            history = data.iloc[:position].copy()
            yield (
                history,
                pd.Timestamp(history.iloc[-1]["date"]),
                pd.Timestamp(data.at[position, "date"]),
                float(data.at[position, "daily_volatility"]),
            )

    attempts: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for completed, result_rows in enumerate(executor.map(_forecast_origin, tasks(), chunksize=1), start=1):
            attempts.extend(result_rows)
            if completed % 25 == 0 or completed == len(test_positions):
                print(f"Completed {completed}/{len(test_positions)} origins.", flush=True)

    transformed = pd.DataFrame(attempts)
    baseline = baseline.rename(columns={"actual": "baseline_actual"})
    forecasts = transformed.merge(
        baseline[
            [
                "target_date",
                "model",
                "baseline_forecast",
                "baseline_actual",
                "baseline_source",
            ]
        ],
        on=["target_date", "model"],
        how="left",
        validate="one_to_one",
    )
    if forecasts["baseline_forecast"].isna().any():
        raise ValueError("Some transformed forecasts have no baseline counterpart.")
    if not np.allclose(
        forecasts["actual"].to_numpy(dtype=float),
        forecasts["baseline_actual"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("Transformed forecasts do not match baseline actuals.")
    forecasts = forecasts.drop(columns="baseline_actual")
    metrics, grouped, q5, dm = _comparison_tables(forecasts)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(OUTPUT_DIR / "log_transform_forecasts.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "log_transform_metrics.csv", index=False)
    grouped.to_csv(OUTPUT_DIR / "log_transform_quintile_metrics.csv", index=False)
    q5.to_csv(OUTPUT_DIR / "log_transform_q5_variance.csv", index=False)
    dm.to_csv(OUTPUT_DIR / "log_transform_dm_tests.csv", index=False)
    _write_report(forecasts, metrics, q5, dm)

    print(metrics.loc[:, ["model_label", "variant", "n", "MAE", "RMSE", "mean_error_actual_minus_forecast"]].to_string(index=False))
    print("\nQ5 same-date variance comparison:")
    print(q5.to_string(index=False))
    print(
        f"\nSaved log-target experiment outputs to "
        f"{OUTPUT_DIR.relative_to(ROOT)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
