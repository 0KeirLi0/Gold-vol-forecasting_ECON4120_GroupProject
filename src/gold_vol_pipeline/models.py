"""Gold-return-only baseline forecasts for forward realized variance."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional for chart generation
    plt = None


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
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate prediction charts. "
            'Install the optional interactive dependencies with "python -m pip install -e \".[interactive]\"".'
        )

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
