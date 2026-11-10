"""Gold-return-only baseline forecasts for forward realized variance."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar


def _validated_returns(returns: Sequence[float] | pd.Series) -> np.ndarray:
    try:
        values = np.asarray(returns, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("returns must contain finite numeric values.") from error
    if values.ndim != 1 or values.size == 0:
        raise ValueError("returns must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain finite numeric values.")
    return values


def _validated_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of observations.")
    return horizon


def expanding_variance_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Forecast horizon-summed variance using all observed squared returns."""
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    return float(horizon * np.mean(np.square(values)))


def rolling_variance_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1, window: int = 20
) -> float:
    """Forecast horizon-summed variance using the most recent squared returns."""
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer number of observations.")
    return float(horizon * np.mean(np.square(values[-window:])))


def ewma_variance_forecast(
    returns: Sequence[float] | pd.Series,
    *,
    horizon: int = 1,
    decay: float = 0.94,
) -> float:
    """Forecast horizon-summed variance with an exponentially weighted estimate."""
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    if not 0 < decay < 1:
        raise ValueError("decay must be strictly between zero and one.")

    squared = np.square(values)
    variance = float(squared[0])
    for observation in squared[1:]:
        variance = decay * variance + (1 - decay) * float(observation)
    return float(horizon * variance)


def ar1_squared_return_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Forecast variance by fitting an AR(1) model to squared returns."""
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    squared = np.square(values)
    if len(squared) < 3:
        raise ValueError("AR(1) squared-return forecasts require at least 3 returns.")

    lagged, observed = squared[:-1], squared[1:]
    lagged_centered = lagged - lagged.mean()
    denominator = float(np.dot(lagged_centered, lagged_centered))
    coefficient = (
        float(np.dot(lagged_centered, observed - observed.mean()) / denominator)
        if denominator > 0
        else 0.0
    )
    coefficient = float(np.clip(coefficient, 0, 1))
    intercept = max(float(observed.mean() - coefficient * lagged.mean()), 0.0)

    forecast = float(squared[-1])
    total = 0.0
    for _ in range(horizon):
        forecast = max(intercept + coefficient * forecast, 0.0)
        total += forecast
    return total


def ma1_squared_return_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Forecast squared returns with a conditional-sum-of-squares MA(1) model."""
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    squared = np.square(values)
    if len(squared) < 3:
        raise ValueError("MA(1) squared-return forecasts require at least 3 returns.")

    mean = float(squared.mean())

    def sum_squared_errors(coefficient: float) -> float:
        previous_error = 0.0
        total = 0.0
        for observation in squared:
            error = float(observation) - mean - coefficient * previous_error
            total += error * error
            previous_error = error
        return total

    fit = minimize_scalar(
        sum_squared_errors, bounds=(-0.99, 0.99), method="bounded"
    )
    if not fit.success or not np.isfinite(fit.x):
        raise RuntimeError(f"MA(1) parameter estimation failed: {fit.message}")

    coefficient = float(fit.x)
    previous_error = 0.0
    for observation in squared:
        previous_error = (
            float(observation) - mean - coefficient * previous_error
        )
    first_forecast = max(mean + coefficient * previous_error, 0.0)
    return float(first_forecast + (horizon - 1) * mean)


def _arch_garch_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int, garch: bool
) -> float:
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    squared = np.square(values)
    if len(squared) < 3:
        model_name = "GARCH(1,1)" if garch else "ARCH(1)"
        raise ValueError(f"{model_name} forecasts require at least 3 returns.")
    if not np.any(squared):
        return 0.0

    sample_variance = max(float(squared.mean()), np.finfo(float).tiny)
    floor = max(sample_variance * 1e-8, np.finfo(float).tiny)
    parameters_count = 3 if garch else 2
    initial = (
        [sample_variance * 0.1, 0.08, 0.82]
        if garch
        else [sample_variance * 0.9, 0.08]
    )
    upper_omega = max(sample_variance * 100, floor * 10)

    def conditional_variances(parameters: np.ndarray) -> np.ndarray:
        omega, alpha = parameters[:2]
        beta = parameters[2] if garch else 0.0
        variance_path = np.empty_like(squared)
        variance_path[0] = sample_variance
        for position in range(1, len(squared)):
            variance_path[position] = (
                omega
                + alpha * squared[position - 1]
                + beta * variance_path[position - 1]
            )
        return np.maximum(variance_path, floor)

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        variance_path = conditional_variances(parameters)
        likelihood = np.log(variance_path[1:]) + squared[1:] / variance_path[1:]
        result = 0.5 * float(np.sum(likelihood))
        return result if np.isfinite(result) else np.finfo(float).max

    constraints = (
        [
            {
                "type": "ineq",
                "fun": lambda parameters: 0.999 - parameters[1] - parameters[2],
            }
        ]
        if garch
        else [{"type": "ineq", "fun": lambda parameters: 0.999 - parameters[1]}]
    )
    fit = minimize(
        negative_log_likelihood,
        np.asarray(initial[:parameters_count]),
        method="SLSQP",
        bounds=[(floor, upper_omega)] + [(0.0, 0.999)] * (parameters_count - 1),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not fit.success or not np.isfinite(fit.fun):
        model_name = "GARCH(1,1)" if garch else "ARCH(1)"
        raise RuntimeError(f"{model_name} parameter estimation failed: {fit.message}")

    omega, alpha = fit.x[:2]
    beta = fit.x[2] if garch else 0.0
    variance_path = conditional_variances(fit.x)
    next_variance = omega + alpha * squared[-1] + beta * variance_path[-1]
    persistence = alpha + beta
    total = max(float(next_variance), floor)
    for _ in range(1, horizon):
        next_variance = omega + persistence * next_variance
        total += max(float(next_variance), floor)
    return float(total)


def arch1_variance_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Fit a Gaussian ARCH(1) model and forecast summed future variance."""
    return _arch_garch_forecast(returns, horizon=horizon, garch=False)


def garch11_variance_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Fit a Gaussian GARCH(1,1) model and forecast summed future variance."""
    return _arch_garch_forecast(returns, horizon=horizon, garch=True)


def baseline_forecasts(
    returns: Sequence[float] | pd.Series,
    *,
    horizon: int = 1,
    rolling_window: int = 20,
    ewma_decay: float = 0.94,
) -> dict[str, float]:
    """Run the available return-only baselines at one forecast origin.

    ``returns`` must end at the forecast origin; future observations must not
    be included. Each value forecasts the sum of variance over ``horizon``
    future observations.
    """
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    return {
        "historical_variance": expanding_variance_forecast(
            values, horizon=horizon
        ),
        "rolling_variance": rolling_variance_forecast(
            values, horizon=horizon, window=rolling_window
        ),
        "ewma_variance": ewma_variance_forecast(
            values, horizon=horizon, decay=ewma_decay
        ),
        "ar1_squared_return": ar1_squared_return_forecast(
            values, horizon=horizon
        ),
        "ma1_squared_return": ma1_squared_return_forecast(
            values, horizon=horizon
        ),
        "arch1": arch1_variance_forecast(values, horizon=horizon),
        "garch11": garch11_variance_forecast(values, horizon=horizon),
    }
