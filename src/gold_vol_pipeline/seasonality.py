"""Seasonality diagnostics for the original LSEG GCc1 Yang-Zhang target."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kruskal


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def build_seasonality_data(prepared: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar fields to valid Yang-Zhang volatility observations."""
    required = {"date", "daily_volatility"}
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"Prepared target is missing columns: {sorted(missing)}")
    if prepared["date"].duplicated().any():
        raise ValueError("Prepared target must contain unique dates.")

    result = prepared.loc[prepared["daily_volatility"].notna()].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["daily_volatility"] = pd.to_numeric(
        result["daily_volatility"], errors="raise"
    )
    if result.empty:
        raise ValueError("Prepared target has no valid volatility observations.")
    if not np.isfinite(result["daily_volatility"].to_numpy()).all():
        raise ValueError("Volatility observations must be finite.")
    if (result["daily_volatility"] < 0).any():
        raise ValueError("Volatility observations must be nonnegative.")

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["month_name"] = result["date"].dt.month.map(
        lambda value: MONTH_NAMES[int(value) - 1]
    )
    result["weekday"] = result["date"].dt.day_name()
    result["weekday"] = pd.Categorical(
        result["weekday"], categories=WEEKDAY_NAMES, ordered=True
    )
    result["lag1_volatility"] = result["daily_volatility"].shift(1)
    return result.sort_values("date").reset_index(drop=True)


def summarize_calendar_groups(
    observations: pd.DataFrame, group: str
) -> pd.DataFrame:
    """Summarize mean and median volatility by month or weekday."""
    if group == "month":
        column = "month_name"
        order = MONTH_NAMES
    elif group == "weekday":
        column = "weekday"
        order = WEEKDAY_NAMES
    else:
        raise ValueError("group must be 'month' or 'weekday'.")

    rows: list[dict[str, float | int | str]] = []
    for name in order:
        values = observations.loc[
            observations[column].astype(str).eq(name), "daily_volatility"
        ].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        rows.append(
            {
                "calendar_group": name,
                "n": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _wald_test(model, positions: Sequence[int]) -> tuple[float, float]:
    params = model.params
    restriction = np.zeros((len(positions), len(params)))
    for row, position in enumerate(positions):
        restriction[row, position] = 1.0
    test = model.wald_test(restriction)
    statistic = float(np.asarray(test.statistic).squeeze())
    pvalue = float(np.asarray(test.pvalue).squeeze())
    return statistic, pvalue


def run_seasonality_tests(
    observations: pd.DataFrame,
    hac_lags: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Run HAC joint category tests and sensitivity checks."""
    lags = tuple(sorted(set(int(lag) for lag in hac_lags)))
    if not lags or lags[0] < 1:
        raise ValueError("HAC lags must be positive.")
    if len(observations) < max(lags) + 30:
        raise ValueError("Too few observations for the requested HAC lag.")

    rows: list[dict[str, float | int | str]] = []
    for lag in lags:
        for category in ("month", "weekday"):
            for include_lag in (False, True):
                if category == "month":
                    dummies = pd.get_dummies(
                        observations["month"],
                        prefix="month",
                        drop_first=True,
                        dtype=float,
                    )
                else:
                    weekday_codes = observations["weekday"].cat.codes
                    dummies = pd.get_dummies(
                        weekday_codes,
                        prefix="weekday",
                        drop_first=True,
                        dtype=float,
                    )
                    dummies.columns = [
                        f"weekday_{WEEKDAY_NAMES[int(column.split('_')[-1])]}"
                        for column in dummies.columns
                    ]
                exog = pd.concat(
                    [
                        pd.Series(
                            1.0, index=observations.index, name="intercept"
                        ),
                        dummies,
                    ],
                    axis=1,
                )
                if include_lag:
                    exog = pd.concat(
                        [
                            exog,
                            observations["lag1_volatility"].rename(
                                "lag1_volatility"
                            ),
                        ],
                        axis=1,
                    )
                valid = exog.notna().all(axis=1)
                exog = exog.loc[valid]
                y = observations.loc[valid, "daily_volatility"]
                model = sm.OLS(y, exog).fit(
                    cov_type="HAC", cov_kwds={"maxlags": lag}
                )
                positions = [
                    exog.columns.get_loc(column) for column in dummies.columns
                ]
                statistic, pvalue = _wald_test(model, positions)
                rows.append(
                    {
                        "category": category,
                        "model": (
                            f"{category}_and_lag1"
                            if include_lag
                            else f"{category}_only"
                        ),
                        "hac_lag": lag,
                        "n": int(len(y)),
                        "wald_chi2": statistic,
                        "p_value": pvalue,
                    }
                )

    for category in ("month", "weekday"):
        column = "month" if category == "month" else "weekday"
        groups = [
            group["daily_volatility"].dropna().to_numpy(dtype=float)
            for _, group in observations.groupby(column, observed=True)
        ]
        statistic, pvalue = kruskal(*groups)
        rows.append(
            {
                "category": category,
                "model": f"{category}_kruskal_wallis",
                "hac_lag": 0,
                "n": int(len(observations)),
                "wald_chi2": float(statistic),
                "p_value": float(pvalue),
            }
        )
    return pd.DataFrame(rows)


def create_seasonality_figures(
    observations: pd.DataFrame,
    month_summary: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    figure_dir: str | Path,
) -> dict[str, str]:
    """Create month, weekday, and month-by-year seasonality figures."""
    figure_root = Path(figure_dir)
    figure_root.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(month_summary["calendar_group"], month_summary["mean"], color="#1f77b4")
    ax.set_title("Mean Yang-Zhang volatility by month")
    ax.set_ylabel("Unannualized daily volatility")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_root / "month_means.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures["month_means"] = str(path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(
        weekday_summary["calendar_group"],
        weekday_summary["mean"],
        color="#d62728",
    )
    ax.set_title("Mean Yang-Zhang volatility by weekday")
    ax.set_ylabel("Unannualized daily volatility")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_root / "weekday_means.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures["weekday_means"] = str(path)

    heatmap = observations.pivot_table(
        index="year",
        columns="month",
        values="daily_volatility",
        aggfunc="mean",
    )
    heatmap = heatmap.reindex(columns=range(1, 13))
    fig, ax = plt.subplots(
        figsize=(11, max(5.0, 0.28 * len(heatmap)))
    )
    image = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(12), MONTH_NAMES)
    ax.set_yticks(range(len(heatmap)), heatmap.index)
    ax.set_title("Monthly mean Yang-Zhang volatility by year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.colorbar(image, ax=ax, label="Unannualized daily volatility")
    fig.tight_layout()
    path = figure_root / "month_year_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures["month_year_heatmap"] = str(path)
    return figures
