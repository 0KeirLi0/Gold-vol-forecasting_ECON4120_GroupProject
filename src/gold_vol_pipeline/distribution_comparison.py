"""Normal versus Student-t rolling-origin comparison for Yang-Zhang forecasts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm, t
from sklearn.ensemble import HistGradientBoostingRegressor

from .volatility_forecasting import (
    EXOGENOUS_COLUMNS,
    SELECTED_PREDICTORS,
    _lagged_design,
    _target,
    _variance_recursion,
)

CANDIDATE_MODELS = ("garch", "egarch", "garch_x", "egarch_x", "har_x", "hgb")
DISTRIBUTIONS = ("normal", "student_t")
MIN_DF = 2.05
MAX_DF = 100.0
MIN_VARIANCE = 1e-12


def _standard_logpdf(
    standardized_errors: np.ndarray,
    distribution: str,
    degrees_freedom: float | None = None,
) -> np.ndarray:
    errors = np.asarray(standardized_errors, dtype=float)
    if distribution == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.square(errors))
    if distribution != "student_t" or degrees_freedom is None:
        raise ValueError(f"Unsupported distribution: {distribution}")
    df = float(degrees_freedom)
    return (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * np.log(np.pi * (df - 2))
        - ((df + 1) / 2) * np.log1p(np.square(errors) / (df - 2))
    )


def _fit_zero_centered_density(
    errors: np.ndarray, distribution: str
) -> tuple[float, float | None]:
    """Estimate a zero-centered unit-variance Normal or Student-t scale."""
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        raise ValueError("At least 10 calibration residuals are required.")
    if distribution == "normal":
        scale = float(np.sqrt(np.mean(np.square(values))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Normal calibration scale must be finite and positive.")
        return scale, None
    if distribution != "student_t":
        raise ValueError(f"Unsupported distribution: {distribution}")

    scale_start = max(float(np.sqrt(np.mean(np.square(values)))), 1e-8)

    def objective(parameters: np.ndarray) -> float:
        scale = float(np.exp(parameters[0]))
        df = float(parameters[1])
        standardized = values / scale
        log_density = (
            _standard_logpdf(standardized, distribution, df) - np.log(scale)
        )
        score = -float(np.sum(log_density))
        return score if np.isfinite(score) else np.finfo(float).max

    fit = minimize(
        objective,
        np.array([np.log(scale_start), 8.0]),
        method="L-BFGS-B",
        bounds=[(-20, 5), (MIN_DF, MAX_DF)],
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Student-t density fit failed: {fit.message}")
    return float(np.exp(fit.x[0])), float(fit.x[1])


def _fit_arch_model(
    history: pd.DataFrame, model: str, distribution: str
) -> dict[str, float | None | str]:
    values = _target(history) * 100
    if len(values) < 100:
        raise ValueError("GARCH and EGARCH require at least 100 observations.")
    volatility = "GARCH" if model == "garch" else "EGARCH"
    fit = arch_model(
        values,
        mean="AR",
        lags=1,
        vol=volatility,
        p=1,
        o=1 if model == "egarch" else 0,
        q=1,
        dist="normal" if distribution == "normal" else "t",
        rescale=False,
    ).fit(disp="off", show_warning=False)
    if int(getattr(fit, "convergence_flag", 1)) != 0:
        raise RuntimeError(f"{model} {distribution} fit did not converge.")
    forecast = fit.forecast(horizon=1)
    mean = float(forecast.mean.iloc[-1, 0]) / 100
    variance = float(forecast.variance.iloc[-1, 0]) / 10_000
    if not np.isfinite([mean, variance]).all() or variance <= 0:
        raise RuntimeError(f"{model} {distribution} produced an invalid forecast.")
    df = float(fit.params["nu"]) if distribution == "student_t" else None
    return {
        "forecast": mean,
        "scale": float(np.sqrt(variance)),
        "degrees_freedom": df,
        "density_space": "level",
        "log_location": np.nan,
    }


def _student_expected_abs(df: float) -> float:
    return float(
        np.exp(
            0.5 * np.log(df - 2)
            + gammaln((df - 1) / 2)
            - 0.5 * np.log(np.pi)
            - gammaln(df / 2)
        )
    )


def _fit_x_arch_model(
    history: pd.DataFrame, model: str, distribution: str
) -> dict[str, float | None | str]:
    """Fit GARCH-X or EGARCH-X with lagged range and selected predictors."""
    values = _target(history) * 100
    if len(values) < 100:
        raise ValueError("GARCH-X and EGARCH-X require 100 observations.")
    range_values = pd.to_numeric(
        history["high_low_log_range"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(range_values).all() or (range_values < 0).any():
        raise ValueError("OHLC range must be finite and non-negative.")
    range_scaled = range_values / max(float(np.mean(range_values)), 1e-12)
    predictors = []
    for column in SELECTED_PREDICTORS:
        values_column = pd.to_numeric(history[column], errors="raise").to_numpy(
            dtype=float
        )
        if not np.isfinite(values_column).all() or (values_column <= 0).any():
            raise ValueError(f"{column} must be finite and positive.")
        predictors.append(values_column / float(np.mean(values_column)))
    exogenous = np.column_stack([range_scaled, *predictors])
    residual_scale = float(np.std(np.diff(values), ddof=1))
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("Conditional-variance model requires varying targets.")
    initial_variance = max(float(np.var(values, ddof=1)) / residual_scale**2, 1e-12)
    is_egarch = model == "egarch_x"
    is_t = distribution == "student_t"
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # GARCH-X: c, phi, omega, alpha, beta, range-X, selected-X.
    # EGARCH-X adds a signed-shock coefficient and uses log variance.
    base_size = 7 if is_egarch else 6
    predictor_count = exogenous.shape[1]
    parameter_count = base_size + predictor_count - 1 + int(is_t)

    def get_df(parameters: np.ndarray) -> float | None:
        return float(parameters[-1]) if is_t else None

    def paths(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        intercept, ar = parameters[:2]
        residuals = np.zeros(len(values), dtype=float)
        residuals[1:] = (
            values[1:] - intercept - ar * values[:-1]
        ) / residual_scale
        variance = np.empty(len(values), dtype=float)
        variance[0] = initial_variance
        if is_egarch:
            omega, alpha, beta, gamma = parameters[2:6]
            x_coefficients = parameters[6 : 6 + predictor_count]
            df = get_df(parameters)
            expected_abs = (
                np.sqrt(2 / np.pi)
                if df is None
                else _student_expected_abs(df)
            )
            for position in range(1, len(values)):
                previous_z = residuals[position - 1] / np.sqrt(
                    max(variance[position - 1], MIN_VARIANCE)
                )
                log_variance = (
                    omega
                    + beta * np.log(max(variance[position - 1], MIN_VARIANCE))
                    + alpha * (abs(previous_z) - expected_abs)
                    + gamma * previous_z
                    + float(exogenous[position - 1] @ x_coefficients)
                )
                # Reject explosive paths instead of clipping them into a fake likelihood.
                if not np.isfinite(log_variance) or abs(log_variance) > 20:
                    variance[:] = np.nan
                    return residuals, variance
                variance[position] = np.exp(log_variance)
        else:
            omega, alpha, beta = parameters[2:5]
            x_coefficients = parameters[5 : 5 + predictor_count]
            forcing = (
                omega
                + alpha * residuals[:-1] ** 2
                + exogenous[:-1] @ x_coefficients
            )
            variance = _variance_recursion(
                forcing, float(beta), initial_variance
            )
        return residuals, variance

    def objective(parameters: np.ndarray) -> float:
        residuals, variance = paths(parameters)
        if (
            not np.isfinite(variance).all()
            or np.any(variance[1:] <= MIN_VARIANCE)
        ):
            return np.finfo(float).max
        standardized = residuals[1:] / np.sqrt(variance[1:])
        df = get_df(parameters)
        log_density = _standard_logpdf(standardized, distribution, df)
        score = -float(
            np.sum(log_density - 0.5 * np.log(variance[1:]))
        )
        return score if np.isfinite(score) else np.finfo(float).max

    variance_start = max(initial_variance * 0.05, 1e-8)
    level_cap = max(float(np.max(values)) * 3, 1.0)
    if is_egarch:
        long_run_variance = max(initial_variance, 1e-6)
        base = [
            float(np.mean(values)),
            0.2,
            (1 - 0.9) * np.log(long_run_variance),
            0.05,
            0.9,
            0.0,
            *([0.0] * predictor_count),
        ]
    else:
        base = [
            float(np.mean(values)),
            0.1,
            variance_start,
            0.05,
            0.85,
            *([0.01] * predictor_count),
        ]
    starts = [np.asarray(base, dtype=float)]
    second = np.asarray(base, dtype=float).copy()
    second[1] = 0.6
    if is_egarch:
        second[3] = 0.1
        second[4] = 0.8
        second[2] = (1 - 0.8) * np.log(max(initial_variance, 1e-6))
    else:
        second[2] *= 2
    starts.append(second)
    if is_t:
        starts = [np.append(start, 8.0) for start in starts]

    if is_egarch:
        bounds = [
            (1e-4, level_cap), (-0.999, 0.999), (-8, 8), (0, 2),
            (-0.999, 0.999), (-3, 3),
        ] + [(-3, 3)] * predictor_count
    else:
        bounds = [
            (None, None), (-0.999, 0.999), (1e-8, 100), (0, 0.999),
            (0, 0.999),
        ] + [(0, 100)] * predictor_count
    if is_t:
        bounds.append((MIN_DF, MAX_DF))
    constraints = (
        ()
        if is_egarch
        else ({"type": "ineq", "fun": lambda parameters: 0.999 - parameters[3] - parameters[4]},)
    )
    fits = [
        minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        for start in starts
    ]
    valid = [
        fit
        for fit in fits
        if fit.success
        and np.isfinite(fit.fun)
        and fit.fun < np.finfo(float).max / 10
        and np.isfinite(fit.x).all()
        and (is_egarch or fit.x[2] > 0)
        and 0 < fit.x[0] + fit.x[1] * values[-1] <= 10 * np.max(values)
    ]
    if not valid:
        messages = "; ".join(str(fit.message) for fit in fits)
        raise RuntimeError(f"{model} {distribution} fit failed: {messages}")
    fit = min(valid, key=lambda result: result.fun)
    parameters = fit.x
    residuals, variance = paths(parameters)
    if is_egarch:
        omega, alpha, beta, gamma = parameters[2:6]
        x_coefficients = parameters[6 : 6 + predictor_count]
        df = get_df(parameters)
        expected_abs = (
            np.sqrt(2 / np.pi) if df is None else _student_expected_abs(df)
        )
        previous_z = residuals[-1] / np.sqrt(max(variance[-1], MIN_VARIANCE))
        next_log_variance = (
            omega
            + beta * np.log(max(variance[-1], MIN_VARIANCE))
            + alpha * (abs(previous_z) - expected_abs)
            + gamma * previous_z
            + float(exogenous[-1] @ x_coefficients)
        )
        if not np.isfinite(next_log_variance) or abs(next_log_variance) > 20:
            raise RuntimeError(f"{model} {distribution} variance path is explosive.")
        next_variance = float(np.exp(next_log_variance))
    else:
        omega, alpha, beta = parameters[2:5]
        x_coefficients = parameters[5 : 5 + predictor_count]
        df = get_df(parameters)
        next_variance = float(
            omega
            + alpha * residuals[-1] ** 2
            + beta * variance[-1]
            + float(exogenous[-1] @ x_coefficients)
        )
    forecast = float((parameters[0] + parameters[1] * values[-1]) / 100)
    forecast_scale = float(residual_scale * np.sqrt(next_variance) / 100)
    max_level = float(np.max(values) / 100)
    if (
        not np.isfinite([forecast, forecast_scale]).all()
        or forecast <= 0
        or forecast_scale <= 0
        or forecast > 10 * max_level
        or forecast_scale > 10 * max_level
    ):
        raise RuntimeError(f"{model} {distribution} produced invalid forecasts.")
    return {
        "forecast": forecast,
        "scale": forecast_scale,
        "degrees_freedom": df,
        "density_space": "level",
        "log_location": np.nan,
    }


def _fit_linear_location(
    features: np.ndarray,
    targets: np.ndarray,
    forecast_features: np.ndarray,
    distribution: str,
) -> dict[str, float | None | str]:
    mean = np.mean(features, axis=0)
    scale_features = np.std(features, axis=0)
    scale_features[scale_features < 1e-12] = 1.0
    x = np.column_stack(
        [np.ones(len(features)), (features - mean) / scale_features]
    )
    x_forecast = np.r_[1.0, ((forecast_features - mean) / scale_features).ravel()]
    coefficients, *_ = np.linalg.lstsq(x, targets, rcond=None)
    residuals = targets - x @ coefficients
    if distribution == "normal":
        error_scale = float(np.sqrt(np.mean(np.square(residuals))))
        df = None
        fitted_coefficients = coefficients
    elif distribution == "student_t":
        start = np.r_[coefficients, np.log(max(np.sqrt(np.mean(residuals**2)), 1e-8)), 8.0]

        def objective(parameters: np.ndarray) -> float:
            residual = targets - x @ parameters[:-2]
            error_scale = float(np.exp(parameters[-2]))
            df = float(parameters[-1])
            log_density = (
                _standard_logpdf(residual / error_scale, "student_t", df)
                - np.log(error_scale)
            )
            score = -float(np.sum(log_density))
            return score if np.isfinite(score) else np.finfo(float).max

        bounds = [(None, None)] * x.shape[1] + [(-20, 5), (MIN_DF, MAX_DF)]
        fit = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if not fit.success or not np.isfinite(fit.fun):
            raise RuntimeError(f"HAR-X Student-t fit failed: {fit.message}")
        fitted_coefficients = fit.x[:-2]
        error_scale = float(np.exp(fit.x[-2]))
        df = float(fit.x[-1])
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    if not np.isfinite(error_scale) or error_scale <= 0:
        raise RuntimeError("HAR-X produced an invalid innovation scale.")
    log_location = float(x_forecast @ fitted_coefficients)
    return {
        "forecast": float(np.exp(log_location)),
        "scale": error_scale,
        "degrees_freedom": df,
        "density_space": "log",
        "log_location": log_location,
    }


def _fit_hgb_locations(
    features: np.ndarray, targets: np.ndarray, forecast_features: np.ndarray
) -> dict[str, dict[str, float | None | str]]:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        random_state=42,
    ).fit(features, targets)
    log_location = float(model.predict(forecast_features)[0])

    calibration_count = min(252, max(50, int(np.floor(len(targets) * 0.2))))
    if len(targets) - calibration_count < 50:
        calibration_count = max(10, len(targets) // 5)
    split = len(targets) - calibration_count
    calibration_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        random_state=42,
    ).fit(features[:split], targets[:split])
    calibration_errors = targets[split:] - calibration_model.predict(features[split:])

    result: dict[str, dict[str, float | None | str]] = {}
    for distribution in DISTRIBUTIONS:
        scale, df = _fit_zero_centered_density(calibration_errors, distribution)
        result[distribution] = {
            "forecast": float(np.exp(log_location)),
            "scale": scale,
            "degrees_freedom": df,
            "density_space": "log",
            "log_location": log_location,
        }
    return result


def _candidate_forecasts(
    history: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], dict[str, float | None | str]],
    dict[tuple[str, str], str],
]:
    """Fit every candidate, keeping a failure from discarding the others."""
    outputs: dict[tuple[str, str], dict[str, float | None | str]] = {}
    errors: dict[tuple[str, str], str] = {}

    def record(model: str, distribution: str, fit):
        try:
            outputs[(model, distribution)] = fit()
        except Exception as error:
            errors[(model, distribution)] = f"{type(error).__name__}: {error}"

    for model in ("garch", "egarch"):
        for distribution in DISTRIBUTIONS:
            record(
                model,
                distribution,
                lambda model=model, distribution=distribution: _fit_arch_model(
                    history, model, distribution
                ),
            )
    for model in ("garch_x", "egarch_x"):
        for distribution in DISTRIBUTIONS:
            record(
                model,
                distribution,
                lambda model=model, distribution=distribution: _fit_x_arch_model(
                    history, model, distribution
                ),
            )

    try:
        features, targets, forecast_features = _lagged_design(
            history,
            include_ohlc=True,
            external_predictors=SELECTED_PREDICTORS,
            target_transform="log",
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        for model in ("har_x", "hgb"):
            for distribution in DISTRIBUTIONS:
                errors[(model, distribution)] = message
        return outputs, errors

    for distribution in DISTRIBUTIONS:
        record(
            "har_x",
            distribution,
            lambda distribution=distribution: _fit_linear_location(
                features, targets, forecast_features, distribution
            ),
        )
    try:
        hgb = _fit_hgb_locations(features, targets, forecast_features)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        for distribution in DISTRIBUTIONS:
            errors[("hgb", distribution)] = message
    else:
        for distribution in DISTRIBUTIONS:
            outputs[("hgb", distribution)] = hgb[distribution]
    return outputs, errors


def _density_log_score(row: pd.Series) -> float:
    if row["density_space"] == "log":
        error = float(np.log(row["actual"]) - row["log_location"])
        jacobian = np.log(row["actual"])
    else:
        error = float(row["actual"] - row["forecast"])
        jacobian = 0.0
    standardized = error / float(row["scale"])
    log_density = float(
        _standard_logpdf(
            np.array([standardized]),
            str(row["distribution"]),
            (
                float(row["degrees_freedom"])
                if pd.notna(row["degrees_freedom"])
                else None
            ),
        )[0]
        - np.log(float(row["scale"]))
        - jacobian
    )
    return -log_density


def _interval(row: pd.Series, coverage: float) -> tuple[float, float]:
    alpha = (1 - coverage) / 2
    if row["distribution"] == "normal":
        quantiles = norm.ppf([alpha, 1 - alpha])
    else:
        df = float(row["degrees_freedom"])
        quantiles = t.ppf([alpha, 1 - alpha], df) * np.sqrt((df - 2) / df)
    scale = float(row["scale"])
    if row["density_space"] == "log":
        location = float(row["log_location"])
        bounds = np.exp(location + scale * quantiles)
    else:
        location = float(row["forecast"])
        bounds = location + scale * quantiles
    return float(bounds[0]), float(bounds[1])


def _blank_origin_row(
    origin: pd.Timestamp,
    target_date: pd.Timestamp,
    model: str,
    distribution: str,
    actual: float,
    history_count: int,
    error: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "origin_date": origin,
        "target_date": target_date,
        "model": model,
        "distribution": distribution,
        "forecast": np.nan,
        "actual": actual,
        "scale": np.nan,
        "degrees_freedom": np.nan,
        "density_space": "",
        "log_location": np.nan,
        "history_count": history_count,
        "converged": False,
        "error": error,
        "negative_log_score": np.nan,
    }
    for coverage in (80, 90, 95):
        row[f"interval_{coverage}_lower"] = np.nan
        row[f"interval_{coverage}_upper"] = np.nan
        row[f"interval_{coverage}_covered"] = False
    return row


def _forecast_origin(
    task: tuple[int, pd.DataFrame, pd.Timestamp, pd.Timestamp, float]
) -> list[dict[str, object]]:
    history_count, history, origin, target_date, actual = task
    predictions, errors = _candidate_forecasts(history)
    rows = []
    for model in CANDIDATE_MODELS:
        for distribution in DISTRIBUTIONS:
            key = (model, distribution)
            if key in errors or key not in predictions:
                rows.append(
                    _blank_origin_row(
                        origin,
                        target_date,
                        model,
                        distribution,
                        actual,
                        history_count,
                        errors.get(key, "missing forecast"),
                    )
                )
                continue
            result = predictions[key]
            try:
                forecast = float(result["forecast"])
                scale = float(result["scale"])
                if (
                    forecast <= 0
                    or scale <= 0
                    or not np.isfinite([forecast, scale]).all()
                ):
                    raise RuntimeError("produced an invalid forecast or scale")
                row = {
                    "origin_date": origin,
                    "target_date": target_date,
                    "model": model,
                    "distribution": distribution,
                    "forecast": forecast,
                    "actual": actual,
                    "scale": scale,
                    "degrees_freedom": result["degrees_freedom"],
                    "density_space": result["density_space"],
                    "log_location": result["log_location"],
                    "history_count": history_count,
                    "converged": True,
                    "error": "",
                }
                row["negative_log_score"] = _density_log_score(pd.Series(row))
                for coverage in (0.8, 0.9, 0.95):
                    lower, upper = _interval(pd.Series(row), coverage)
                    label = int(coverage * 100)
                    row[f"interval_{label}_lower"] = lower
                    row[f"interval_{label}_upper"] = upper
                    row[f"interval_{label}_covered"] = lower <= actual <= upper
            except Exception as error:
                rows.append(
                    _blank_origin_row(
                        origin,
                        target_date,
                        model,
                        distribution,
                        actual,
                        history_count,
                        f"{type(error).__name__}: {error}",
                    )
                )
            else:
                rows.append(row)
    return rows


def evaluate_distribution_comparison(
    data: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    workers: int = 1,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run expanding-history one-step distribution comparison."""
    required = {
        "date",
        "daily_volatility",
        "high_low_log_range",
        *EXOGENOUS_COLUMNS,
        *SELECTED_PREDICTORS,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if workers < 1:
        raise ValueError("workers must be positive.")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.dropna(subset=["daily_volatility"]).sort_values("date")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Target dates must be nonmissing and unique.")
    for column in (*SELECTED_PREDICTORS, "high_low_log_range", *EXOGENOUS_COLUMNS):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite for every target date.")
    split = int(np.floor(len(frame) * train_fraction))
    if split < 100 or split >= len(frame):
        raise ValueError("Chronological split leaves too few training or test rows.")

    tasks = [
        (
            position,
            frame.iloc[:position].copy(),
            pd.Timestamp(frame.iloc[position - 1]["date"]),
            pd.Timestamp(frame.iloc[position]["date"]),
            float(frame.iloc[position]["daily_volatility"]),
        )
        for position in range(split, len(frame))
    ]
    output: list[list[dict[str, object]]] = []
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            output.append(_forecast_origin(task))
            if progress and (index == 1 or index % 25 == 0 or index == len(tasks)):
                print(f"Completed {index}/{len(tasks)} forecast origins.", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, rows in enumerate(
                executor.map(_forecast_origin, tasks, chunksize=1), start=1
            ):
                output.append(rows)
                if progress and (
                    index == 1 or index % 25 == 0 or index == len(tasks)
                ):
                    print(
                        f"Completed {index}/{len(tasks)} forecast origins.",
                        flush=True,
                    )
    results = pd.DataFrame([row for rows in output for row in rows])
    results["forecast_error"] = results["forecast"] - results["actual"]
    results["squared_error"] = np.square(results["forecast_error"])
    summary = summarize_distribution_comparison(results)
    manifest = {
        "instrument": "LSEG/Refinitiv GCc1 generic COMEX gold futures",
        "target": "unannualized square root of two-observation Yang-Zhang variance",
        "train_fraction": train_fraction,
        "training_observations": split,
        "test_observations": len(frame) - split,
        "test_start": frame.iloc[split]["date"].strftime("%Y-%m-%d"),
        "test_end": frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "models": list(CANDIDATE_MODELS),
        "distributions": list(DISTRIBUTIONS),
        "predictors": {
            "GARCH-X and EGARCH-X": [
                "lagged high_low_log_range",
                *SELECTED_PREDICTORS,
            ],
            "HAR-X and HGB": [*EXOGENOUS_COLUMNS, *SELECTED_PREDICTORS],
        },
        "hgb_distribution_note": (
            "HGB point model remains squared-error gradient boosting under both "
            "assumptions; Normal and Student-t predictive densities use errors "
            "from a chronological training-only calibration slice."
        ),
        "student_t_note": (
            "Student-t innovations are standardized to unit variance and degrees "
            "of freedom are estimated by maximum likelihood with df > 2."
        ),
        "garch_x_note": (
            "GARCH-X and EGARCH-X are re-estimated in this experiment under each "
            "likelihood. GARCH-X keeps the existing AR(1) mean, lagged OHLC range, "
            "nonnegative selected-predictor coefficients, and Gaussian special case. "
            "It is not the earlier threshold-GARCH variant. EGARCH-X is new."
        ),
        "conditional_variance_note": (
            "A heavier-tailed innovation does not impose constant variance of "
            "forecast errors across forecast levels. Quintile error dispersion "
            "remains a descriptive check of that pattern."
        ),
        "outputs": [],
    }
    return results, summary, manifest


def summarize_distribution_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize common-date point and density performance."""
    required = {
        "model",
        "distribution",
        "target_date",
        "actual",
        "forecast",
        "negative_log_score",
        "interval_80_covered",
        "interval_80_lower",
        "interval_80_upper",
        "interval_90_covered",
        "interval_90_lower",
        "interval_90_upper",
        "interval_95_covered",
        "interval_95_lower",
        "interval_95_upper",
        "converged",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    successful = results.loc[
        results["converged"]
        & np.isfinite(results["forecast"])
        & np.isfinite(results["negative_log_score"])
    ].copy()
    dates = successful.groupby(["model", "distribution"])["target_date"].apply(
        lambda values: set(values)
    )
    common_dates = set.intersection(*dates.tolist()) if not dates.empty else set()
    common = successful.loc[successful["target_date"].isin(common_dates)]
    if common.empty:
        raise ValueError("Candidates have no common successful target dates.")

    attempt_counts = results.groupby(["model", "distribution"], sort=False).size()
    failure_counts = (
        results.loc[~results["converged"].astype(bool)]
        .groupby(["model", "distribution"], sort=False)
        .size()
    )
    rows: list[dict[str, float | int | str]] = []
    for (model, distribution), group in common.groupby(
        ["model", "distribution"], sort=False
    ):
        errors = group["forecast"].to_numpy(dtype=float) - group["actual"].to_numpy(
            dtype=float
        )
        actual_var = np.square(group["actual"].to_numpy(dtype=float))
        forecast_var = np.square(group["forecast"].to_numpy(dtype=float))
        ratio = np.maximum(actual_var, np.finfo(float).tiny) / forecast_var
        row: dict[str, float | int | str] = {
            "model": str(model),
            "distribution": str(distribution),
            "attempts": int(attempt_counts.loc[(model, distribution)]),
            "failure_count": int(failure_counts[(model, distribution)])
            if (model, distribution) in failure_counts.index
            else 0,
            "n_common": int(len(group)),
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
            "bias_forecast_minus_actual": float(np.mean(errors)),
            "QLIKE": float(np.mean(ratio - np.log(ratio) - 1)),
            "mean_negative_log_score": float(group["negative_log_score"].mean()),
            "df_median": float(group["degrees_freedom"].median())
            if group["degrees_freedom"].notna().any()
            else np.nan,
        }
        for coverage in (80, 90, 95):
            row[f"coverage_{coverage}"] = float(
                group[f"interval_{coverage}_covered"].mean()
            )
            row[f"mean_width_{coverage}"] = float(
                (
                    group[f"interval_{coverage}_upper"]
                    - group[f"interval_{coverage}_lower"]
                ).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_forecast_quintiles(results: pd.DataFrame) -> pd.DataFrame:
    """Describe raw forecast-error dispersion across within-model forecast bins."""
    successful = results.loc[
        results["converged"] & np.isfinite(results["forecast"])
    ].copy()
    rows: list[dict[str, float | int | str]] = []
    for (model, distribution), group in successful.groupby(
        ["model", "distribution"], sort=False
    ):
        group = group.sort_values("forecast").copy()
        group["forecast_quintile"] = pd.qcut(
            np.arange(len(group)), q=5, labels=False, duplicates="drop"
        ) + 1
        for quintile, block in group.groupby("forecast_quintile"):
            errors = block["actual"].to_numpy(dtype=float) - block[
                "forecast"
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "model": str(model),
                    "distribution": str(distribution),
                    "forecast_quintile": int(quintile),
                    "n": int(len(block)),
                    "mean_forecast": float(block["forecast"].mean()),
                    "mean_actual": float(block["actual"].mean()),
                    "mean_error_actual_minus_forecast": float(np.mean(errors)),
                    "error_sd": float(np.std(errors, ddof=1)),
                    "error_variance": float(np.var(errors, ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def save_distribution_comparison(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    quintiles: pd.DataFrame,
    manifest: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist all experiment tables and a provenance manifest."""
    import hashlib
    import json

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": output / "distribution_forecasts.csv",
        "summary": output / "distribution_summary.csv",
        "quintiles": output / "forecast_error_quintiles.csv",
        "manifest": output / "distribution_manifest.json",
    }
    results.to_csv(paths["forecasts"], index=False)
    summary.to_csv(paths["summary"], index=False)
    quintiles.to_csv(paths["quintiles"], index=False)
    payload = dict(manifest)
    payload["outputs"] = {name: str(path) for name, path in paths.items()}
    if "input_path" in payload:
        input_path = Path(str(payload["input_path"]))
        payload["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    paths["manifest"].write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return paths

