"""Gold-return-only baseline forecasts for forward realized variance."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from arch import arch_model
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter


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
        [sample_variance * 0.05, 0.05, 0.90]
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
    if not fit.success or not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        model_name = "GARCH(1,1)" if garch else "ARCH(1)"
        raise RuntimeError(f"{model_name} parameter estimation failed: {fit.message}")

    omega, alpha = fit.x[:2]
    beta = fit.x[2] if garch else 0.0
    if garch and (
        omega < floor * (1 - 1e-6)
        or omega > upper_omega * (1 + 1e-6)
        or alpha < -1e-8
        or beta < -1e-8
        or alpha + beta > 0.999 + 1e-8
    ):
        raise RuntimeError("GARCH(1,1) parameter estimation violated constraints.")
    variance_path = conditional_variances(fit.x)
    next_variance = omega + alpha * squared[-1] + beta * variance_path[-1]
    persistence = alpha + beta
    total = max(float(next_variance), floor)
    for _ in range(1, horizon):
        next_variance = omega + persistence * next_variance
        total += max(float(next_variance), floor)
    if not np.isfinite(total) or total < 0:
        raise RuntimeError("ARCH/GARCH model produced an invalid variance forecast.")
    return float(total)


def arch1_variance_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Fit a Gaussian ARCH(1) model and forecast summed future variance."""
    return _arch_garch_forecast(returns, horizon=horizon, garch=False)


def garch11_variance_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Fit scaled Gaussian GARCH(1,1) and forecast variance in original units."""
    values = _validated_returns(returns)
    _validated_horizon(horizon)
    if len(values) < 3:
        raise ValueError("GARCH(1,1) forecasts require at least 3 returns.")
    scale = float(np.sqrt(np.mean(np.square(values))))
    if not np.isfinite(scale):
        raise ValueError("GARCH(1,1) return scale must be finite.")
    if scale == 0:
        if np.any(values):
            raise ValueError("GARCH(1,1) return scale underflowed.")
        return 0.0
    forecast = _arch_garch_forecast(values / scale, horizon=horizon, garch=True)
    result = forecast * scale**2
    if not np.isfinite(result) or result <= 0:
        raise RuntimeError("GARCH(1,1) produced an invalid rescaled forecast.")
    return result


def _arch_package_forecast(
    returns: Sequence[float] | pd.Series,
    *,
    horizon: int,
    vol: str = "GARCH",
    p: int = 1,
    o: int = 0,
    q: int = 1,
    dist: str = "normal",
    simulation: bool = False,
) -> float:
    """Fit an arch volatility model to scaled returns and sum forecast variances."""
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    if len(values) < 30:
        raise ValueError(f"{vol} forecasts require at least 30 returns.")
    if not np.any(values):
        return 0.0

    fit = arch_model(
        values * 100,
        mean="Constant",
        vol=vol,
        p=p,
        o=o,
        q=q,
        dist=dist,
        rescale=False,
    ).fit(disp="off", show_warning=False)
    convergence_flag = int(getattr(fit, "convergence_flag", 1))
    if convergence_flag != 0:
        raise RuntimeError(
            f"{vol} model did not converge (arch convergence flag "
            f"{convergence_flag})."
        )

    forecast_options: dict[str, object] = {}
    if simulation:
        forecast_options = {
            "method": "simulation",
            "simulations": 1000,
            "rng": np.random.default_rng(42).standard_normal,
        }
    variance = fit.forecast(horizon=horizon, **forecast_options).variance.iloc[-1]
    forecast = float(variance.sum()) / 10_000
    if not np.isfinite(forecast) or forecast < 0:
        raise RuntimeError(f"{vol} model produced an invalid variance forecast.")
    return forecast


def gjr_garch_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Fit a Gaussian asymmetric GJR-GARCH(1,1,1) and sum future variance."""
    return _arch_package_forecast(
        returns, horizon=horizon, vol="GARCH", p=1, o=1, q=1
    )


def egarch_asym_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Fit asymmetric EGARCH(1,1,1), using simulation for multi-step forecasts."""
    return _arch_package_forecast(
        returns,
        horizon=horizon,
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        simulation=horizon > 1,
    )


def garch_t_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Fit a Student-t GARCH(1,1) and sum future variance."""
    return _arch_package_forecast(
        returns, horizon=horizon, vol="GARCH", p=1, o=0, q=1, dist="t"
    )


def garchx_forecast(
    returns: Sequence[float] | pd.Series,
    predictors: pd.DataFrame,
    *,
    horizon: int = 1,
) -> float:
    """Fit zero-mean GARCH-X with lagged VIX, holding VIX fixed in forecasts."""
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    if not isinstance(predictors, pd.DataFrame) or "vix_close" not in predictors:
        raise ValueError("predictors must include a 'vix_close' column.")
    if isinstance(returns, pd.Series) and not returns.index.equals(predictors.index):
        raise ValueError("GARCH-X returns and predictors must have matching indexes.")
    vix = np.asarray(predictors["vix_close"], dtype=float)
    if len(values) < 100 or len(vix) != len(values):
        raise ValueError("GARCH-X requires at least 100 aligned returns and VIX values.")
    if not np.isfinite(vix).all() or np.any(vix <= 0):
        raise ValueError("GARCH-X requires finite positive VIX values.")
    scale = float(np.sqrt(np.mean(np.square(values))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("GARCH-X requires a finite nonzero return scale.")
    squared = np.square(values / scale)
    vix_scaled = vix / float(vix.mean())

    def path(parameters: np.ndarray) -> np.ndarray:
        omega, alpha, beta, delta = parameters
        inputs = omega + alpha * squared[:-1] + delta * vix_scaled[:-1]
        later, _ = lfilter([1.0], [1.0, -beta], inputs, zi=[beta])
        return np.concatenate(([1.0], later))

    def objective(parameters: np.ndarray) -> float:
        variance = path(parameters)[1:]
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            return np.finfo(float).max
        return float(0.5 * np.sum(np.log(variance) + squared[1:] / variance))

    constraint = {"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]}
    fits = [
        minimize(
            objective, start, method="SLSQP",
            bounds=[(1e-8, 100), (0, 0.999), (0, 0.999), (0, 100)],
            constraints=[constraint], options={"maxiter": 1000, "ftol": 1e-10},
        )
        for start in ([0.04, 0.05, 0.85, 0.05], [0.1, 0.05, 0.8, 0.05])
    ]
    valid = [
        fit for fit in fits if fit.success and np.isfinite(fit.fun)
        and np.isfinite(fit.x).all() and fit.x[0] >= 1e-8 * (1 - 1e-6)
        and fit.x[0] <= 100 * (1 + 1e-6)
        and np.all(fit.x[1:] >= -1e-8)
        and fit.x[1] + fit.x[2] <= 0.999 + 1e-8
        and fit.x[3] <= 100 * (1 + 1e-6)
    ]
    if not valid:
        raise RuntimeError(
            "GARCH-X parameter estimation failed or violated constraints: "
            + "; ".join(str(fit.message) for fit in fits)
        )
    fit = min(valid, key=lambda result: result.fun)
    omega, alpha, beta, delta = fit.x
    next_variance = (
        omega + alpha * squared[-1] + beta * path(fit.x)[-1]
        + delta * vix_scaled[-1]
    )
    total = float(next_variance)
    for _ in range(1, horizon):
        next_variance = omega + (alpha + beta) * next_variance + delta * vix_scaled[-1]
        total += float(next_variance)
    forecast = total * scale**2
    if not np.isfinite(forecast) or forecast <= 0:
        raise RuntimeError("GARCH-X produced an invalid variance forecast.")
    return forecast


def ewma_estimated_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> float:
    """Select EWMA decay by in-window QLIKE over a fixed 0.80–0.99 grid."""
    values = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    squared = np.square(values)
    if not np.any(squared):
        return 0.0
    if len(squared) < 3:
        raise ValueError("Estimated EWMA forecasts require at least 3 returns.")

    decay_grid = np.linspace(0.80, 0.99, 20)
    epsilon = max(float(squared.mean()) * 1e-12, np.finfo(float).tiny)
    best_decay: float | None = None
    best_loss = np.inf
    for decay in decay_grid:
        variance = float(squared[0])
        losses: list[float] = []
        for observed in squared[1:]:
            forecast = max(variance, epsilon)
            actual = max(float(observed), epsilon)
            ratio = actual / forecast
            losses.append(ratio - np.log(ratio) - 1)
            variance = decay * variance + (1 - decay) * float(observed)
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_decay = float(decay)
    if best_decay is None or not np.isfinite(best_loss):
        raise RuntimeError("EWMA decay selection failed.")

    variance = float(squared[0])
    for observed in squared[1:]:
        variance = best_decay * variance + (1 - best_decay) * float(observed)
    return float(horizon * variance)


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


def _load_returns_from_csv(csv_path: str, *, column: str = "log_return") -> pd.Series:
    """Load a prepared dataset and return the log-return series for forecasting."""
    frame = pd.read_csv(csv_path)
    if column not in frame.columns:
        raise ValueError(
            f"Column '{column}' was not found in '{csv_path}'. "
            f"Available columns: {list(frame.columns)}"
        )
    returns = frame[column].dropna()
    if returns.empty:
        raise ValueError(f"Column '{column}' in '{csv_path}' contains no usable values.")
    return returns


def _print_forecasts(forecasts: dict[str, float]) -> None:
    """Format and print the forecast dictionary in a human-readable way."""
    for model_name, value in forecasts.items():
        print(f"{model_name}: {float(value):.12f}")


def prediction_history(
    csv_path: str,
    *,
    horizon: int = 5,
    rolling_window: int = 20,
    ewma_decay: float = 0.94,
    column: str = "log_return",
    target_column: str = "realized_variance",
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Backtest each baseline on the prepared data and compare forecasts to actuals."""
    frame = pd.read_csv(csv_path)
    if sample_size is not None:
        if (
            isinstance(sample_size, bool)
            or not isinstance(sample_size, int)
            or sample_size < 1
        ):
            raise ValueError("sample_size must be a positive integer or None.")
        first_origin = max(0, len(frame) - sample_size)
    else:
        first_origin = 0
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' not found in '{csv_path}'")
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in '{csv_path}'")

    rows: list[dict[str, object]] = []
    for index in range(first_origin, len(frame)):
        origin_returns = frame.loc[:index, column].dropna()
        if len(origin_returns) < 3:
            continue
        try:
            forecast = baseline_forecasts(
                origin_returns,
                horizon=horizon,
                rolling_window=rolling_window,
                ewma_decay=ewma_decay,
            )
        except (RuntimeError, ValueError):
            continue
        row = {
            "date": frame.iloc[index]["date"],
            "actual": float(frame.iloc[index][target_column]),
        }
        row.update(forecast)
        rows.append(row)

    if not rows:
        raise ValueError("No prediction rows were generated from the supplied dataset.")
    return pd.DataFrame(rows)


def generate_prediction_chart(
    csv_path: str,
    *,
    horizon: int = 5,
    rolling_window: int = 20,
    ewma_decay: float = 0.94,
    output_path: str = "plots/prediction_vs_actual.png",
    sample_size: int | None = 500,
    column: str = "log_return",
    target_column: str = "realized_variance",
) -> str:
    """Save a comparison chart of forecasted versus actual realized variance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "matplotlib is required to generate prediction charts. "
            'Install the optional interactive dependencies with "python -m pip install -e \".[interactive]\"".'
        ) from error

    comparison = prediction_history(
        csv_path,
        horizon=horizon,
        rolling_window=rolling_window,
        ewma_decay=ewma_decay,
        column=column,
        target_column=target_column,
        sample_size=sample_size,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(comparison["date"], comparison["actual"], label="actual", color="black", linewidth=2)
    for model_name in [
        "historical_variance",
        "rolling_variance",
        "ewma_variance",
        "ar1_squared_return",
        "ma1_squared_return",
        "arch1",
        "garch11",
    ]:
        axis.plot(comparison["date"], comparison[model_name], label=model_name, alpha=0.8)
    axis.set_title("Forecasted vs actual realized variance")
    axis.set_ylabel("Realized variance")
    axis.set_xlabel("Forecast origin date")
    axis.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return str(output)


def main() -> int:
    """Run a forecast on a prepared CSV and print the prediction results."""
    parser = argparse.ArgumentParser(
        description="Forecast forward realized variance from a prepared log-return series."
    )
    parser.add_argument(
        "--input",
        default="data/processed/gold_futures.csv",
        help="CSV file containing the prepared dataset.",
    )
    parser.add_argument(
        "--column",
        default="log_return",
        help="Column with log returns to use for the forecast.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Number of future observations to aggregate into the variance forecast.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Window length for the rolling-variance baseline.",
    )
    parser.add_argument(
        "--ewma-decay",
        type=float,
        default=0.94,
        help="Decay parameter for the EWMA baseline.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional output path for a forecast-vs-actual chart. When omitted, no chart is created.",
    )
    args = parser.parse_args()

    returns = _load_returns_from_csv(args.input, column=args.column)
    forecasts = baseline_forecasts(
        returns,
        horizon=args.horizon,
        rolling_window=args.rolling_window,
        ewma_decay=args.ewma_decay,
    )

    print(f"Loaded {len(returns)} return observations from {args.input}")
    print(f"Forecast horizon: {args.horizon}")
    _print_forecasts(forecasts)

    if args.plot_output:
        chart_path = generate_prediction_chart(
            args.input,
            horizon=args.horizon,
            rolling_window=args.rolling_window,
            ewma_decay=args.ewma_decay,
            output_path=args.plot_output,
        )
        print(f"Chart saved to {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
