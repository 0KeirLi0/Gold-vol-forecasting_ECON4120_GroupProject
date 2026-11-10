"""Hyndman-Khandakar stepwise ARIMA selection for volatility proxy series."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.stattools import kpss


def _validated_series(series: Sequence[float] | pd.Series) -> np.ndarray:
    try:
        values = np.asarray(series, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("series must contain finite numeric values.") from error
    if values.ndim != 1 or values.size < 30:
        raise ValueError("series must contain at least 30 one-dimensional observations.")
    if not np.isfinite(values).all():
        raise ValueError("series must contain finite numeric values.")
    if np.all(values == values[0]):
        raise ValueError("series must have non-zero variance for ARIMA selection.")
    return values


@dataclass(frozen=True)
class HyndmanKhandakarResult:
    """Selected nonseasonal ARIMA model and the candidates considered."""

    order: tuple[int, int, int]
    fit: ARIMAResultsWrapper
    candidates: pd.DataFrame
    differencing_p_values: tuple[float, ...]

    @property
    def aicc(self) -> float:
        """Return the selected model's small-sample corrected AIC."""
        selected = self.candidates.loc[
            self.candidates["order"] == str(self.order), "aicc"
        ]
        return float(selected.iloc[0])

    def forecast(self, horizon: int = 5) -> np.ndarray:
        """Forecast nonnegative period-by-period variance proxies."""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        values = np.asarray(self.fit.get_forecast(steps=horizon).predicted_mean)
        return np.maximum(values, 0.0)


def hyndman_khandakar(
    series: Sequence[float] | pd.Series,
    *,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    max_order: int = 5,
    alpha: float = 0.05,
) -> HyndmanKhandakarResult:
    """Select a nonseasonal ARIMA order using KPSS differencing and stepwise AICc.

    This implements the nonseasonal Hyndman-Khandakar procedure: repeated KPSS
    tests choose ``d``, then a bounded stepwise search over ``p`` and ``q``
    minimizes AICc. It is not a seasonal ARIMA implementation.
    """
    values = _validated_series(series)
    integer_options = {
        "max_p": max_p,
        "max_q": max_q,
        "max_d": max_d,
        "max_order": max_order,
    }
    for name, value in integer_options.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")

    differenced = values
    d = 0
    differencing_p_values: list[float] = []
    while True:
        _, p_value, *_ = kpss(differenced, regression="c", nlags="auto")
        differencing_p_values.append(float(p_value))
        if p_value >= alpha or d >= max_d:
            break
        differenced = np.diff(differenced)
        d += 1

    candidate_records: dict[tuple[int, int], dict[str, object]] = {}
    fitted_models: dict[tuple[int, int], ARIMAResultsWrapper] = {}

    def fit_candidate(p: int, q: int) -> float:
        key = (p, q)
        if key in candidate_records:
            return float(candidate_records[key]["aicc"])

        record: dict[str, object] = {
            "order": str((p, d, q)),
            "aic": np.nan,
            "aicc": np.inf,
            "bic": np.nan,
            "status": "failed",
            "message": "",
        }
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = ARIMA(
                    values,
                    order=(p, d, q),
                    trend="c" if d == 0 else "t" if d == 1 else "n",
                ).fit()
            converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
            if not converged:
                record["message"] = "optimizer did not converge"
            else:
                parameter_count = len(fit.params)
                observations = float(fit.nobs)
                correction_denominator = observations - parameter_count - 1
                aic = float(fit.aic)
                aicc = (
                    aic
                    + 2 * parameter_count * (parameter_count + 1)
                    / correction_denominator
                    if correction_denominator > 0
                    else np.inf
                )
                record.update(
                    {
                        "aic": aic,
                        "aicc": float(aicc),
                        "bic": float(fit.bic),
                        "status": "ok",
                    }
                )
                fitted_models[key] = fit
            if caught:
                record["message"] = "; ".join(
                    str(item.message) for item in caught
                )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            record["message"] = str(error)

        candidate_records[key] = record
        return float(record["aicc"])

    def allowed(p: int, q: int) -> bool:
        return p <= max_p and q <= max_q and p + q <= max_order

    seed_orders = {
        (0, 0),
        (min(2, max_p), min(2, max_q)),
        (min(1, max_p), 0),
        (0, min(1, max_q)),
    }
    seed_orders = {order for order in seed_orders if allowed(*order)}
    for p, q in sorted(seed_orders):
        fit_candidate(p, q)

    valid_seeds = [
        key for key, record in candidate_records.items() if record["status"] == "ok"
    ]
    if not valid_seeds:
        raise RuntimeError("No ARIMA candidate converged; inspect the input series.")
    current = min(valid_seeds, key=lambda key: float(candidate_records[key]["aicc"]))

    while True:
        p, q = current
        neighbors = {
            (p + p_step, q + q_step)
            for p_step in (-1, 0, 1)
            for q_step in (-1, 0, 1)
            if (p_step != 0 or q_step != 0)
        }
        neighbors = {
            (candidate_p, candidate_q)
            for candidate_p, candidate_q in neighbors
            if candidate_p >= 0
            and candidate_q >= 0
            and allowed(candidate_p, candidate_q)
        }
        for candidate_p, candidate_q in sorted(neighbors):
            fit_candidate(candidate_p, candidate_q)
        valid_neighbors = [
            key
            for key in neighbors
            if candidate_records.get(key, {}).get("status") == "ok"
        ]
        best_neighbor = min(
            valid_neighbors,
            key=lambda key: float(candidate_records[key]["aicc"]),
            default=current,
        )
        if float(candidate_records[best_neighbor]["aicc"]) >= float(
            candidate_records[current]["aicc"]
        ):
            break
        current = best_neighbor

    candidates = pd.DataFrame(candidate_records.values()).sort_values(
        "aicc", ignore_index=True
    )
    selected_fit = fitted_models[current]
    return HyndmanKhandakarResult(
        order=(current[0], d, current[1]),
        fit=selected_fit,
        candidates=candidates,
        differencing_p_values=tuple(differencing_p_values),
    )


def main() -> int:
    """Fit the selected ARIMA model to returns or squared returns from CSV."""
    parser = argparse.ArgumentParser(
        description="Select and fit a nonseasonal ARIMA model with the "
        "Hyndman-Khandakar stepwise AICc procedure."
    )
    parser.add_argument(
        "--input",
        default="data/processed/gold_futures.csv",
        help="CSV file containing the prepared dataset.",
    )
    parser.add_argument(
        "--column",
        default="log_return",
        help="Column containing the source returns.",
    )
    parser.add_argument(
        "--transform",
        choices=("squared_returns", "returns"),
        default="squared_returns",
        help="Fit the model to squared returns (volatility proxy) or raw returns.",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-p", type=int, default=5)
    parser.add_argument("--max-q", type=int, default=5)
    parser.add_argument("--max-d", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=5)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.column not in frame.columns:
        raise ValueError(
            f"Column '{args.column}' was not found in '{args.input}'. "
            f"Available columns: {list(frame.columns)}"
        )
    source = pd.to_numeric(frame[args.column], errors="raise").dropna().to_numpy()
    series = source**2 if args.transform == "squared_returns" else source
    result = hyndman_khandakar(
        series,
        max_p=args.max_p,
        max_q=args.max_q,
        max_d=args.max_d,
        max_order=args.max_order,
    )
    forecasts = result.forecast(args.horizon)
    selected = result.candidates.loc[
        result.candidates["order"] == str(result.order)
    ].iloc[0]

    print(f"Input: {args.input} [{args.column}; {args.transform}], n={len(series)}")
    print(f"Selected order: ARIMA{result.order}")
    print(f"Selected AICc: {result.aicc:.6f}")
    print(f"KPSS p-values for differencing decisions: {result.differencing_p_values}")
    print("Fitted parameters:")
    for name, value in zip(result.fit.param_names, result.fit.params):
        print(f"  {name}: {float(value):.10g}")
    print(
        f"Forecasted {args.horizon}-step sum of nonnegative "
        f"{args.transform}: {float(forecasts.sum()):.10g}"
    )
    print("\nBest candidate orders by AICc:")
    print(
        result.candidates[["order", "aicc", "status"]]
        .head(10)
        .to_string(index=False)
    )
    if selected["status"] != "ok":
        raise RuntimeError("Selected ARIMA candidate did not converge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
