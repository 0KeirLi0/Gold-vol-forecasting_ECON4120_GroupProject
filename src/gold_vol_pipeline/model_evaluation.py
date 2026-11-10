"""Exploratory time-series diagnostics and in-sample model comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class ModelEvaluation:
    """Fitted candidates and their information criteria and residual series."""

    model_comparison: pd.DataFrame
    fitted_models: dict[str, Any]
    residuals: pd.DataFrame
    standardized_residuals: pd.DataFrame


def _numeric_series(
    values: Sequence[float] | pd.Series, *, min_observations: int = 1
) -> pd.Series:
    try:
        series = values.astype(float).copy() if isinstance(values, pd.Series) else pd.Series(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("series must contain finite numeric values.") from error
    if series.ndim != 1 or len(series) < min_observations:
        raise ValueError(
            f"series must contain at least {min_observations} observations."
        )
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError("series must contain finite numeric values without missing observations.")
    return series


def transform_price_series(prices: Sequence[float] | pd.Series) -> pd.DataFrame:
    """Return price levels, logs, differences, returns, and squared returns."""
    series = _numeric_series(prices, min_observations=2)
    if (series <= 0).any():
        raise ValueError("prices must be strictly positive for log transformations.")
    log_price = np.log(series)
    return pd.DataFrame(
        {
            "price": series,
            "log_price": log_price,
            "first_difference": series.diff(),
            "log_difference": log_price.diff(),
            "percent_change": series.pct_change(fill_method=None),
            "squared_log_return": log_price.diff().pow(2),
        },
        index=series.index,
    )


def adf_unit_root_test(
    series: Sequence[float] | pd.Series,
    *,
    name: str = "series",
    regression: str = "c",
    autolag: str | None = "AIC",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an Augmented Dickey-Fuller test and report its unit-root null."""
    values = _numeric_series(series, min_observations=20)
    if regression not in {"c", "ct", "ctt", "n"}:
        raise ValueError("regression must be one of 'c', 'ct', 'ctt', or 'n'.")
    if autolag not in {None, "AIC", "BIC", "t-stat"}:
        raise ValueError("autolag must be None, 'AIC', 'BIC', or 't-stat'.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")

    statistic, p_value, used_lag, nobs, critical_values, *_ = adfuller(
        values.to_numpy(), regression=regression, autolag=autolag
    )
    significance_levels = {"1%": 0.01, "5%": 0.05, "10%": 0.10}
    critical_level = min(
        significance_levels,
        key=lambda level: abs(significance_levels[level] - alpha),
    )
    critical_column = f"critical_value_{critical_level}"
    critical_value = critical_values[critical_level]
    return pd.DataFrame(
        [
            {
                "series": name,
                "adf_statistic": statistic,
                "p_value": p_value,
                "used_lag": used_lag,
                "nobs": nobs,
                critical_column: critical_value,
                "reject_unit_root": bool(p_value < alpha),
                "null_hypothesis": "The series has a unit root.",
            }
        ]
    )


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            'Plotting requires matplotlib; install with "python -m pip install -e \".[interactive]\"".'
        ) from error
    return plt


def plot_series_transformations(
    transformed: pd.DataFrame,
    *,
    columns: Sequence[str] = (
        "price",
        "log_price",
        "first_difference",
        "log_difference",
        "squared_log_return",
    ),
    output_path: str | Path | None = None,
):
    """Plot selected level and transformed series in a shared-x panel."""
    missing = [column for column in columns if column not in transformed]
    if missing:
        raise ValueError(f"transformed data are missing columns: {missing}")
    plt = _pyplot()
    figure, axes = plt.subplots(
        len(columns), 1, figsize=(12, max(3, 2.4 * len(columns))), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, column in zip(axes, columns, strict=True):
        axis.plot(transformed.index, transformed[column], linewidth=0.8)
        axis.set_ylabel(column.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[0].set_title("Original price series and transformations")
    axes[-1].set_xlabel("Date / observation")
    figure.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    return figure, axes


def plot_acf_pacf(
    series: Sequence[float] | pd.Series,
    *,
    name: str = "series",
    lags: int = 40,
    output_path: str | Path | None = None,
):
    """Plot ACF and PACF to inspect serial dependence and persistence."""
    values = _numeric_series(series, min_observations=5)
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer.")
    effective_lags = min(lags, len(values) // 2 - 1)
    if effective_lags < 1:
        raise ValueError("series is too short for the requested ACF/PACF plot.")
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(values, lags=effective_lags, ax=axes[0], zero=False)
    plot_pacf(values, lags=effective_lags, ax=axes[1], zero=False, method="ywm")
    axes[0].set_title(f"ACF: {name}")
    axes[1].set_title(f"PACF: {name}")
    figure.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    return figure, axes


def _positive_orders(
    orders: Sequence[int], *, name: str, allow_zero: bool = False
) -> tuple[int, ...]:
    result = tuple(orders)
    minimum = 0 if allow_zero else 1
    if any(
        isinstance(order, bool) or not isinstance(order, int) or order < minimum
        for order in result
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain {qualifier} integer orders.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate orders.")
    return result


def _positive_order_pairs(
    orders: Sequence[tuple[int, int]], *, name: str
) -> tuple[tuple[int, int], ...]:
    result = tuple(tuple(order) for order in orders)
    if any(
        len(order) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in order
        )
        for order in result
    ):
        raise ValueError(f"{name} must contain pairs of positive integer orders.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate orders.")
    return result


def compare_candidate_models(
    series: Sequence[float] | pd.Series,
    *,
    ar_orders: Sequence[int] = (1, 2, 3),
    ma_orders: Sequence[int] = (1, 2, 3),
    arma_orders: Sequence[tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2)),
    arch_orders: Sequence[int] = (1, 2, 3),
    garch_orders: Sequence[tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2)),
    egarch_orders: Sequence[tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2)),
) -> ModelEvaluation:
    """Fit AR/MA/ARMA and ARCH/GARCH/EGARCH candidates on one series.

    The returned information criteria are in-sample summaries, not evidence
    that a candidate forecasts best out of sample. All candidates use the same
    observations and Gaussian likelihood assumptions where available.
    """
    values = _numeric_series(series, min_observations=30)
    ar_orders = _positive_orders(ar_orders, name="ar_orders")
    ma_orders = _positive_orders(ma_orders, name="ma_orders")
    arch_orders = _positive_orders(arch_orders, name="arch_orders")
    arma_orders = _positive_order_pairs(arma_orders, name="arma_orders")
    garch_orders = _positive_order_pairs(garch_orders, name="garch_orders")
    egarch_orders = _positive_order_pairs(egarch_orders, name="egarch_orders")
    if not any((ar_orders, ma_orders, arma_orders, arch_orders, garch_orders, egarch_orders)):
        raise ValueError("at least one candidate model order must be provided.")

    fitted_models: dict[str, Any] = {}
    residual_series: dict[str, pd.Series] = {}
    standardized_series: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []

    def add_arima_candidate(
        family: str, name: str, order: tuple[int, int, int]
    ) -> None:
        fit = ARIMA(values, order=order, trend="c").fit()
        converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
        residual = pd.Series(np.asarray(fit.resid), index=values.index, name=name)
        scale = float(residual.std(ddof=1))
        standardized = residual / scale if scale > 0 else residual * np.nan
        fitted_models[name] = fit
        residual_series[name] = residual
        standardized_series[name] = standardized
        rows.append(
            {
                "model_name": name,
                "model_family": family,
                "order": str(order),
                "AIC": float(fit.aic),
                "BIC": float(fit.bic),
                "log_likelihood": float(fit.llf),
                "nobs": int(fit.nobs),
                "converged": converged,
            }
        )

    def add_volatility_candidate(
        family: str, name: str, p: int, q: int = 0
    ) -> None:
        fit = arch_model(
            values,
            mean="Constant",
            vol=family,
            p=p,
            q=q,
            dist="normal",
            rescale=False,
        ).fit(disp="off")
        converged = fit.convergence_flag == 0
        residual = pd.Series(np.asarray(fit.resid), index=values.index, name=name)
        standardized = pd.Series(
            np.asarray(fit.std_resid), index=values.index, name=name
        )
        fitted_models[name] = fit
        residual_series[name] = residual
        standardized_series[name] = standardized
        rows.append(
            {
                "model_name": name,
                "model_family": family,
                "order": f"({p}, {q})",
                "AIC": float(fit.aic),
                "BIC": float(fit.bic),
                "log_likelihood": float(fit.loglikelihood),
                "nobs": int(fit.nobs),
                "converged": converged,
            }
        )

    for p in ar_orders:
        add_arima_candidate("AR", f"AR({p})", (p, 0, 0))
    for q in ma_orders:
        add_arima_candidate("MA", f"MA({q})", (0, 0, q))
    for p, q in arma_orders:
        add_arima_candidate("ARMA", f"ARMA({p},{q})", (p, 0, q))
    for p in arch_orders:
        add_volatility_candidate("ARCH", f"ARCH({p})", p)
    for p, q in garch_orders:
        add_volatility_candidate("GARCH", f"GARCH({p},{q})", p, q)
    for p, q in egarch_orders:
        add_volatility_candidate("EGARCH", f"EGARCH({p},{q})", p, q)

    comparison = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    residuals = pd.DataFrame(residual_series, index=values.index)
    standardized_residuals = pd.DataFrame(standardized_series, index=values.index)
    return ModelEvaluation(
        model_comparison=comparison,
        fitted_models=fitted_models,
        residuals=residuals,
        standardized_residuals=standardized_residuals,
    )


def ljung_box_residual_tests(
    evaluation: ModelEvaluation,
    *,
    lags: Sequence[int] = (10, 20),
    model_df: int = 0,
) -> pd.DataFrame:
    """Test residual dependence with an optional Ljung-Box model-DF correction.

    The default ``model_df=0`` reports unadjusted p-values; callers may provide
    a correction appropriate to their fitted model and inference plan.
    """
    test_lags = tuple(lags)
    if not test_lags or any(
        isinstance(lag, bool) or not isinstance(lag, int) or lag < 1
        for lag in test_lags
    ):
        raise ValueError("lags must contain positive integer values.")
    if len(set(test_lags)) != len(test_lags):
        raise ValueError("lags must not contain duplicates.")
    if isinstance(model_df, bool) or not isinstance(model_df, int) or model_df < 0:
        raise ValueError("model_df must be a nonnegative integer.")

    rows: list[dict[str, object]] = []
    for model_name in evaluation.residuals.columns:
        series_by_kind = {
            "residuals": evaluation.residuals[model_name].dropna(),
            "squared_standardized_residuals": (
                evaluation.standardized_residuals[model_name].dropna().pow(2)
            ),
        }
        for kind, residuals in series_by_kind.items():
            valid_lags = [lag for lag in test_lags if lag < len(residuals)]
            if not valid_lags:
                raise ValueError(
                    f"residual series for {model_name} is too short for lags {test_lags}."
                )
            result = acorr_ljungbox(
                residuals,
                lags=valid_lags,
                model_df=model_df,
                return_df=True,
            )
            for lag, row in result.iterrows():
                rows.append(
                    {
                        "model_name": model_name,
                        "series_tested": kind,
                        "lag": int(lag),
                        "model_df": model_df,
                        "ljung_box_statistic": float(row["lb_stat"]),
                        "p_value": float(row["lb_pvalue"]),
                    }
                )
    return pd.DataFrame(rows)


def save_residuals(
    evaluation: ModelEvaluation,
    output_path: str | Path,
    *,
    standardized: bool = False,
) -> str:
    """Save raw or standardized model residuals to a date-indexed CSV."""
    residuals = (
        evaluation.standardized_residuals if standardized else evaluation.residuals
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(output, index_label=residuals.index.name or "date")
    return str(output)


def plot_residual_panel(
    evaluation: ModelEvaluation,
    *,
    standardized: bool = False,
    output_path: str | Path | None = None,
    columns: int = 2,
):
    """Plot each fitted candidate's residual time series in a panel."""
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer.")
    residuals = (
        evaluation.standardized_residuals if standardized else evaluation.residuals
    )
    plt = _pyplot()
    model_names = list(residuals.columns)
    rows = (len(model_names) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(14, max(3.2, 2.7 * rows)), sharex=True
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, model_name in zip(axes, model_names, strict=False):
        axis.plot(residuals.index, residuals[model_name], linewidth=0.7)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(model_name)
        axis.grid(alpha=0.2)
    for axis in axes[len(model_names) :]:
        axis.set_visible(False)
    axes[0].set_ylabel(
        "Standardized residual" if standardized else "Residual"
    )
    figure.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    return figure, axes
