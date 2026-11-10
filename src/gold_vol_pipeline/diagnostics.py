"""Statistical diagnostics for return data used by the volatility baselines."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, ttest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def _validated_returns(returns: Sequence[float] | pd.Series) -> np.ndarray:
    try:
        values = np.asarray(returns, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("returns must contain finite numeric values.") from error
    if values.ndim != 1 or values.size < 20:
        raise ValueError("returns must contain at least 20 one-dimensional observations.")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain finite numeric values.")
    if np.all(values == values[0]):
        raise ValueError("returns must have non-zero variance for statistical diagnostics.")
    return values


def check_return_assumptions(
    returns: Sequence[float] | pd.Series, *, lags: int = 10, alpha: float = 0.05
) -> pd.DataFrame:
    """Run exploratory checks relevant to the project's return-only baselines.

    The tests identify evidence against selected assumptions; passing a test
    does not establish that a model is correctly specified.
    """
    values = _validated_returns(returns)
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")

    test_lags = min(lags, max(1, len(values) // 5))
    rows: list[dict[str, object]] = []

    def add_test(
        name: str,
        statistic: float,
        p_value: float,
        null_hypothesis: str,
        reject_interpretation: str,
        retain_interpretation: str,
    ) -> None:
        reject = p_value < alpha
        rows.append(
            {
                "diagnostic": name,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "result": "evidence against null" if reject else "no evidence against null",
                "interpretation": (
                    reject_interpretation if reject else retain_interpretation
                ),
                "null_hypothesis": null_hypothesis,
            }
        )

    mean_test = ttest_1samp(values, popmean=0)
    add_test(
        "mean_zero_t_test",
        mean_test.statistic,
        mean_test.pvalue,
        "The unconditional mean return is zero.",
        "The mean differs from zero; consider centering returns before squared-return models.",
        "The sample does not provide evidence that the mean differs from zero.",
    )

    for label, series in (("returns", values), ("squared_returns", values**2)):
        adf_lags = min(lags, max(0, len(series) // 2 - 2))
        adf_stat, adf_p_value, *_ = adfuller(
            series, maxlag=adf_lags, regression="c", autolag="AIC"
        )
        add_test(
            f"{label}_adf_stationarity",
            adf_stat,
            adf_p_value,
            "The series has a unit root.",
            "Rejecting the unit-root null supports stationarity.",
            "The test does not reject a unit root; stationarity is not established.",
        )

    for label, series in (("returns", values), ("squared_returns", values**2)):
        result = acorr_ljungbox(series, lags=[test_lags], return_df=True)
        add_test(
            f"{label}_ljung_box",
            result["lb_stat"].iloc[0],
            result["lb_pvalue"].iloc[0],
            f"No autocorrelation through lag {test_lags}.",
            "Serial dependence is detected at the tested lags.",
            "No serial dependence is detected at the tested lags.",
        )

    arch_stat, arch_p_value, _, _ = het_arch(values, nlags=test_lags)
    add_test(
        "arch_lm",
        arch_stat,
        arch_p_value,
        f"No ARCH effects through lag {test_lags}.",
        "Conditional heteroskedasticity is detected; ARCH/GARCH-style models may be relevant.",
        "No ARCH effects are detected at the tested lags.",
    )

    jb = jarque_bera(values)
    add_test(
        "return_normality_jarque_bera",
        jb.statistic,
        jb.pvalue,
        "Returns are normally distributed.",
        "Returns depart from normality; Gaussian likelihood forecasts should be interpreted cautiously.",
        "The test does not detect a departure from normality.",
    )

    summary = pd.DataFrame(
        [
            {
                "diagnostic": "sample_summary",
                "statistic": np.nan,
                "p_value": np.nan,
                "result": "descriptive",
                "interpretation": (
                    f"n={len(values)}, mean={np.mean(values):.6g}, "
                    f"std={np.std(values, ddof=1):.6g}, "
                    f"skew={pd.Series(values).skew():.4g}, "
                    f"excess_kurtosis={pd.Series(values).kurtosis():.4g}"
                ),
                "null_hypothesis": "",
            }
        ]
    )
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)


def main() -> int:
    """Print model-relevant diagnostics for a prepared CSV return column."""
    parser = argparse.ArgumentParser(
        description="Check statistical properties of returns used by volatility baselines."
    )
    parser.add_argument(
        "--input",
        default="data/processed/gold_futures.csv",
        help="CSV file containing the prepared dataset.",
    )
    parser.add_argument(
        "--column",
        default="log_return",
        help="Column containing returns to diagnose.",
    )
    parser.add_argument("--lags", type=int, default=10, help="Maximum test lag.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Test significance level.")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.column not in frame.columns:
        raise ValueError(
            f"Column '{args.column}' was not found in '{args.input}'. "
            f"Available columns: {list(frame.columns)}"
        )
    returns = pd.to_numeric(frame[args.column], errors="raise").dropna()
    report = check_return_assumptions(returns, lags=args.lags, alpha=args.alpha)
    print(f"Diagnostics for {args.input} [{args.column}], n={len(returns)}")
    print(report.to_string(index=False))
    print(
        "\nInterpretation is diagnostic, not a model-approval rule. "
        "Failure to reject a null hypothesis does not prove it is true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
