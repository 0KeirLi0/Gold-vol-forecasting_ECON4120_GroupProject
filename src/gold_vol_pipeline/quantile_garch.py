"""Exploratory GARCH model with an extra response to extreme returns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from .models import _validated_horizon, _validated_returns


@dataclass(frozen=True)
class QuantileGarchFit:
    """Fitted coefficients and the state at the last observed return."""

    omega: float
    alpha: float
    beta: float
    gamma: float
    threshold: float
    last_return: float
    last_variance: float
    observations: int
    extreme_observations: int
    negative_log_likelihood_scaled: float

    def forecast(
        self, *, horizon: int = 1, simulations: int = 20_000, seed: int = 42
    ) -> float:
        """Sum future conditional variances using Gaussian simulation after step one."""
        horizon = _validated_horizon(horizon)
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1:
            raise ValueError("simulations must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        shock = self.last_return**2
        first = (
            self.omega + self.alpha * shock + self.beta * self.last_variance
            + self.gamma * shock * (abs(self.last_return) > self.threshold)
        )
        if not np.isfinite(first) or first <= 0:
            raise RuntimeError("Quantile GARCH produced an invalid first variance.")
        if horizon == 1:
            return float(first)
        rng = np.random.default_rng(seed)
        variances = np.full(simulations, first)
        total = first
        for _ in range(1, horizon):
            returns = np.sqrt(variances) * rng.standard_normal(simulations)
            squared = np.square(returns)
            variances = (
                self.omega + self.alpha * squared + self.beta * variances
                + self.gamma * squared * (np.abs(returns) > self.threshold)
            )
            total += float(variances.mean())
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Quantile GARCH produced an invalid variance forecast.")
        return float(total)


def fit_quantile_garch(
    returns: Sequence[float] | pd.Series, *, quantile: float = 0.95
) -> QuantileGarchFit:
    """Fit a zero-mean Gaussian GARCH with a fixed in-sample tail threshold."""
    values = _validated_returns(returns)
    if len(values) < 100:
        raise ValueError("Quantile GARCH requires at least 100 returns.")
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one.")
    scale = float(np.sqrt(np.mean(np.square(values))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Quantile GARCH requires a finite, nonzero return scale.")
    scaled = values / scale
    squared = np.square(scaled)
    threshold = float(np.quantile(np.abs(scaled), quantile))
    extreme_squared = squared * (np.abs(scaled) > threshold)
    if not np.any(extreme_squared[:-1]):
        raise ValueError("Quantile GARCH needs at least one extreme lagged return.")
    floor = 1e-8

    def variances(parameters: np.ndarray) -> np.ndarray:
        omega, alpha, beta, gamma = parameters
        inputs = omega + alpha * squared[:-1] + gamma * extreme_squared[:-1]
        later, _ = lfilter([1.0], [1.0, -beta], inputs, zi=[beta])
        return np.concatenate(([1.0], later))

    def objective(parameters: np.ndarray) -> float:
        path = variances(parameters)
        if not np.isfinite(path).all() or np.any(path <= 0):
            return np.finfo(float).max
        return float(0.5 * np.sum(np.log(path[1:]) + squared[1:] / path[1:]))

    constraint = {"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2] - x[3]}
    fits = [
        minimize(
            objective, start, method="SLSQP",
            bounds=[(floor, 100.0), (0.0, 0.999), (0.0, 0.999), (0.0, 0.999)],
            constraints=[constraint], options={"maxiter": 1000, "ftol": 1e-10},
        )
        for start in ([0.05, 0.04, 0.89, 0.02], [0.20, 0.04, 0.70, 0.03])
    ]
    valid = [
        fit for fit in fits
        if fit.success and np.isfinite(fit.fun) and np.isfinite(fit.x).all()
        and floor * (1 - 1e-6) <= fit.x[0] <= 100.0 * (1 + 1e-6)
        and np.all(fit.x[1:] >= -1e-8) and np.all(fit.x[1:] <= 0.999 + 1e-8)
        and np.sum(fit.x[1:]) <= 0.999 + 1e-8
    ]
    if not valid:
        raise RuntimeError(
            "Quantile GARCH parameter estimation failed or violated constraints: "
            + "; ".join(str(fit.message) for fit in fits)
        )
    fit = min(valid, key=lambda result: result.fun)
    omega, alpha, beta, gamma = (float(value) for value in fit.x)
    last_variance = float(variances(fit.x)[-1] * scale**2)
    return QuantileGarchFit(
        omega=omega * scale**2, alpha=alpha, beta=beta, gamma=gamma,
        threshold=threshold * scale, last_return=float(values[-1]),
        last_variance=last_variance, observations=len(values),
        extreme_observations=int(np.count_nonzero(extreme_squared)),
        negative_log_likelihood_scaled=float(fit.fun),
    )
