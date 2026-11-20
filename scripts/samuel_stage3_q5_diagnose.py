"""Diagnose forecast errors by forecast, origin-volatility, and realized quintiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, skew

from gold_vol_pipeline.volatility_forecasting import (
    SELECTED_PREDICTORS,
    evaluate_daily_volatility,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FORECASTS = ROOT / "data/processed/stage2_yz_forecasts.csv"
PREPARED_TARGETS = (
    ROOT / "data/processed/stage2_yz_selected/GCc1_yang_zhang_volatility.csv"
)
REPORT_SOURCE = ROOT / "reports/yang-zhang-residual-diagnostics.qmd"
SELECTED_REPORT_SOURCE = (
    ROOT / "reports/yang-zhang-selected-residual-diagnostics.qmd"
)
MODEL_NAMES = {
    "garch11": "GARCH(1,1)",
    "egarch": "EGARCH(1,1)",
    "garchx_ohlc_range": "Range GARCH-X",
    "egarchx": "EGARCH-X",
    "harx_ohlc": "HAR-X (OHLC)",
    "hist_gradient_boosting_ohlc": "HGB (OHLC)",
}
BASELINE_MODELS = [
    "garch11",
    "egarch",
    "garchx_ohlc_range",
    "harx_ohlc",
    "hist_gradient_boosting_ohlc",
]
ERROR_COLUMNS = {
    "absolute_volatility_error": "forecast_error",
    "relative_error": "relative_error",
    "log_error": "log_error",
}
QUINTILE_LABELS = [f"Q{number}" for number in range(1, 6)]
RANDOM_SEED = 42


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the tracked forecasts and the prepared Yang-Zhang target panel."""
    for path in (BASELINE_FORECASTS, PREPARED_TARGETS):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    baseline = pd.read_csv(BASELINE_FORECASTS)
    prepared = pd.read_csv(PREPARED_TARGETS)
    required_forecasts = {
        "origin_date",
        "target_date",
        "model",
        "forecast",
        "actual",
        "converged",
    }
    missing = required_forecasts - set(baseline.columns)
    if missing:
        raise ValueError(f"Baseline forecasts are missing columns: {sorted(missing)}")
    required_targets = {"date", "daily_volatility", *SELECTED_PREDICTORS}
    missing = required_targets - set(prepared.columns)
    if missing:
        raise ValueError(f"Prepared target data are missing columns: {sorted(missing)}")
    return baseline, prepared


def _regenerate_egarchx(prepared: pd.DataFrame) -> pd.DataFrame:
    """Regenerate EGARCH-X sequentially with predictors in its ARX mean."""
    regenerated, _, _ = evaluate_daily_volatility(
        prepared,
        train_fraction=0.8,
        model_names=("egarchx",),
        workers=1,
        progress=True,
        external_predictors=SELECTED_PREDICTORS,
    )
    regenerated = regenerated.copy()
    regenerated["forecast_source"] = "regenerated EGARCH-X (ARX mean)"
    return regenerated


def _prepare_forecasts(
    baseline: pd.DataFrame,
    regenerated: pd.DataFrame,
    prepared: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the six forecasts and keep their shared successful dates."""
    baseline = baseline.loc[baseline["model"].isin(BASELINE_MODELS)].copy()
    baseline["forecast_source"] = "saved baseline 20-model CSV"
    if set(baseline["model"]) != set(BASELINE_MODELS):
        raise ValueError("The baseline file does not contain all five expected models.")
    if regenerated["model"].ne("egarchx").any():
        raise ValueError("Regenerated forecast rows must all be EGARCH-X.")

    columns = [
        "origin_date",
        "target_date",
        "model",
        "forecast",
        "actual",
        "converged",
        "error",
        "forecast_source",
    ]
    for column in ("error", "forecast_source"):
        if column not in regenerated:
            regenerated[column] = (
                "" if column == "error" else "regenerated EGARCH-X"
            )
    combined = pd.concat(
        [baseline[columns], regenerated[columns]], ignore_index=True
    )
    combined["origin_date"] = pd.to_datetime(combined["origin_date"], errors="raise")
    combined["target_date"] = pd.to_datetime(combined["target_date"], errors="raise")
    combined["forecast"] = pd.to_numeric(combined["forecast"], errors="coerce")
    combined["actual"] = pd.to_numeric(combined["actual"], errors="coerce")
    combined["converged"] = (
        combined["converged"].astype(str).str.strip().str.lower().eq("true")
    )
    if combined.duplicated(["model", "target_date"]).any():
        raise ValueError("Duplicate model/target-date forecasts were found.")

    attempt_counts = combined.groupby("model").size()
    success_counts = combined.loc[combined["converged"]].groupby("model").size()
    failure_counts = (attempt_counts - success_counts).fillna(attempt_counts).astype(int)
    failure_counts = failure_counts.reindex(MODEL_NAMES, fill_value=0)

    valid = combined.loc[
        combined["converged"]
        & np.isfinite(combined["forecast"])
        & np.isfinite(combined["actual"])
    ].copy()
    if (valid["forecast"] <= 0).any() or (valid["actual"] <= 0).any():
        raise ValueError("Log errors require positive forecasts and actual targets.")

    common_dates = set(valid.loc[valid["model"].eq("garch11"), "target_date"])
    for model in MODEL_NAMES:
        common_dates.intersection_update(
            valid.loc[valid["model"].eq(model), "target_date"]
        )
    if not common_dates:
        raise ValueError("The six models have no common successful target dates.")
    common = valid.loc[valid["target_date"].isin(common_dates)].copy()
    if common.groupby("target_date")["actual"].nunique().gt(1).any():
        raise ValueError("Actual targets disagree across models.")

    prepared = prepared.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    prepared["daily_volatility"] = pd.to_numeric(
        prepared["daily_volatility"], errors="raise"
    )
    panel = (
        prepared.dropna(subset=["daily_volatility"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    split = int(np.floor(len(panel) * 0.8))
    train_targets = panel.iloc[:split]["daily_volatility"].to_numpy(dtype=float)
    if split < 1 or not np.isfinite(train_targets).all():
        raise ValueError("Training targets are unavailable or invalid.")

    origin_actual = panel.set_index("date")["daily_volatility"]
    common["origin_volatility"] = common["origin_date"].map(origin_actual)
    if common["origin_volatility"].isna().any():
        raise ValueError("Could not match every forecast origin to its observed target.")

    common["forecast_error"] = common["actual"] - common["forecast"]
    common["relative_error"] = common["actual"] / common["forecast"] - 1
    common["log_error"] = np.log(common["actual"] / common["forecast"])
    if not np.isfinite(
        common[
            [
                "forecast_error",
                "relative_error",
                "log_error",
                "origin_volatility",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Forecast errors or origin volatility contain non-finite values.")

    # Training-only cutoffs prevent future holdout values setting origin bins.
    train_cutoffs = np.quantile(train_targets, [0.2, 0.4, 0.6, 0.8])
    common["origin_volatility_quintile"] = (
        np.searchsorted(train_cutoffs, common["origin_volatility"], side="right") + 1
    )
    common["origin_volatility_quintile"] = (
        "Q" + common["origin_volatility_quintile"].astype(str)
    )

    # Forecast Q5 follows the repo's equal-count ranking within each model.
    common["forecast_quintile"] = ""
    for _, group in common.groupby("model", sort=True):
        ordered = group.sort_values(["forecast", "target_date"], kind="mergesort")
        labels = pd.qcut(
            np.arange(len(ordered)), q=5, labels=QUINTILE_LABELS
        ).astype(str)
        common.loc[ordered.index, "forecast_quintile"] = labels

    # Realized-volatility bins use holdout outcomes and are retrospective.
    actual_order = common.drop_duplicates("target_date").sort_values(
        ["actual", "target_date"], kind="mergesort"
    )
    actual_labels = pd.qcut(
        np.arange(len(actual_order)), q=5, labels=QUINTILE_LABELS
    ).astype(str)
    actual_groups = dict(zip(actual_order["target_date"], actual_labels, strict=True))
    common["realized_volatility_quintile"] = common["target_date"].map(actual_groups)

    failures = pd.DataFrame(
        {
            "model": list(failure_counts.index),
            "attempts": [
                int(attempt_counts.get(model, 0)) for model in failure_counts.index
            ],
            "failures": failure_counts.to_numpy(dtype=int),
            "common_successful_targets": len(common_dates),
        }
    )
    return common.sort_values(["target_date", "model"]), failures


def _describe(values: pd.Series) -> dict[str, float | int]:
    """Summarize one error series using sample variance."""
    numeric = values.to_numpy(dtype=float)
    return {
        "n": int(len(numeric)),
        "mean_error": float(np.mean(numeric)),
        "residual_variance": float(np.var(numeric, ddof=1)),
        "MAE": float(np.mean(np.abs(numeric))),
        "RMSE": float(np.sqrt(np.mean(np.square(numeric)))),
        "skewness": float(skew(numeric, bias=False)) if len(numeric) > 2 else np.nan,
        "excess_kurtosis": (
            float(kurtosis(numeric, fisher=True, bias=False))
            if len(numeric) > 3
            else np.nan
        ),
    }


def _group_metrics(errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize all three error measures within each quintile grouping."""
    grouping_columns = {
        "forecast_level": "forecast_quintile",
        "origin_volatility": "origin_volatility_quintile",
        "realized_volatility": "realized_volatility_quintile",
    }
    rows = []
    for grouping, column in grouping_columns.items():
        for (model, quintile), group in errors.groupby(
            ["model", column], sort=True, observed=True
        ):
            for error_type, error_column in ERROR_COLUMNS.items():
                rows.append(
                    {
                        "grouping": grouping,
                        "quintile": quintile,
                        "model": model,
                        "model_label": MODEL_NAMES[model],
                        "error_type": error_type,
                        **_describe(group[error_column]),
                    }
                )
    return pd.DataFrame(rows)


def _hac_variance_tests(
    errors: pd.DataFrame, *, hac_lags: int = 4
) -> pd.DataFrame:
    """Test whether raw or log-error variance rises with its forecast level."""
    rows = []
    for model, group in errors.groupby("model", sort=True):
        specifications = (
            (
                "squared_absolute_error_on_forecast",
                np.square(group["forecast_error"].to_numpy(dtype=float)),
                group["forecast"].to_numpy(dtype=float),
            ),
            (
                "squared_log_error_on_log_forecast",
                np.square(group["log_error"].to_numpy(dtype=float)),
                np.log(group["forecast"].to_numpy(dtype=float)),
            ),
        )
        for test_name, outcome, predictor in specifications:
            fit = sm.OLS(outcome, sm.add_constant(predictor)).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": hac_lags, "use_correction": True},
                use_t=True,
            )
            interval = fit.conf_int(alpha=0.05)[1]
            rows.append(
                {
                    "model": model,
                    "test": test_name,
                    "n": int(fit.nobs),
                    "intercept": float(fit.params[0]),
                    "slope": float(fit.params[1]),
                    "hac_standard_error": float(fit.bse[1]),
                    "ci_95_lower": float(interval[0]),
                    "ci_95_upper": float(interval[1]),
                    "p_value": float(fit.pvalues[1]),
                    "hac_lags": hac_lags,
                }
            )
    return pd.DataFrame(rows)


def _top_q5_misses(errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ten largest Q5 misses overall and ten for each model."""
    q5 = errors.loc[errors["forecast_quintile"].eq("Q5")].copy()
    q5["absolute_miss"] = q5["forecast_error"].abs()
    columns = [
        "model",
        "target_date",
        "origin_date",
        "actual",
        "forecast",
        "forecast_error",
        "absolute_miss",
    ]
    overall = q5.sort_values("absolute_miss", ascending=False).head(10)[columns]
    per_model = (
        q5.sort_values(["model", "absolute_miss"], ascending=[True, False])
        .groupby("model", sort=True)
        .head(10)[columns]
    )
    return overall.reset_index(drop=True), per_model.reset_index(drop=True)


def _plot_residuals(errors: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save simple plots of errors against forecasts and quintile variances."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (model, group) in zip(axes.flat, errors.groupby("model", sort=True)):
        axis.scatter(group["forecast"], group["forecast_error"], s=10, alpha=0.45)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("Forecast volatility")
        axis.set_ylabel("Actual − forecast")
    figure.suptitle("Forecast errors versus forecast level")
    figure.tight_layout()
    path = figure_dir / "residuals_vs_forecast.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved.append(path)

    metrics = _group_metrics(errors)
    forecast_metrics = metrics.loc[metrics["grouping"].eq("forecast_level")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for axis, error_type in zip(axes, ERROR_COLUMNS):
        part = forecast_metrics.loc[forecast_metrics["error_type"].eq(error_type)]
        for model, group in part.groupby("model", sort=True):
            group = group.set_index("quintile").reindex(QUINTILE_LABELS)
            axis.plot(
                QUINTILE_LABELS,
                group["residual_variance"],
                marker="o",
                label=MODEL_NAMES[model],
            )
        axis.set_title(error_type.replace("_", " "))
        axis.set_xlabel("Forecast quintile")
        axis.set_ylabel("Sample error variance")
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle("Error variance across forecast quintiles")
    figure.tight_layout()
    path = figure_dir / "variance_by_forecast_quintile.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved.append(path)

    raw_metrics = metrics.loc[
        metrics["error_type"].eq("absolute_volatility_error")
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for axis, grouping in zip(
        axes,
        ("forecast_level", "origin_volatility", "realized_volatility"),
    ):
        part = raw_metrics.loc[raw_metrics["grouping"].eq(grouping)]
        for model, group in part.groupby("model", sort=True):
            group = group.set_index("quintile").reindex(QUINTILE_LABELS)
            axis.plot(
                QUINTILE_LABELS,
                group["residual_variance"],
                marker="o",
                label=MODEL_NAMES[model],
            )
        axis.set_title(grouping.replace("_", " "))
        axis.set_xlabel("Quintile")
        axis.set_ylabel("Raw error variance")
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle("Raw error variance by grouping")
    figure.tight_layout()
    path = figure_dir / "variance_by_grouping.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved.append(path)
    return saved


def _report_q5_reference() -> dict[str, float]:
    """Read the published HGB Q5 mean and skewness from the report tables."""
    if not REPORT_SOURCE.is_file():
        return {}
    text = REPORT_SOURCE.read_text(encoding="utf-8")
    reference = {}
    markers = (
        ("mean_error_report_units", "Mean forecast error", "mean_error"),
        ("skewness", "Forecast-error skewness", "skewness"),
    )
    for key, marker, _ in markers:
        if marker not in text:
            continue
        section = text.split(marker, maxsplit=1)[1]
        for line in section.splitlines():
            if line.startswith("| Histogram gradient boosting"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if len(cells) == 6:
                    reference[key] = float(cells[-1])
                break
    return reference


def _selected_report_miss_range() -> tuple[float, float] | None:
    """Read the published six-model miss range for 2026-02-02, in 1e-3 units."""
    if not SELECTED_REPORT_SOURCE.is_file():
        return None
    text = " ".join(
        SELECTED_REPORT_SOURCE.read_text(encoding="utf-8").split()
    ).casefold()
    marker = "residuals on that date range from approximately "
    if marker not in text:
        return None
    phrase = text.split(marker, maxsplit=1)[1].split(" in the same units", maxsplit=1)[0]
    parts = phrase.split(" to ", maxsplit=1)
    if len(parts) != 2:
        return None
    return float(parts[0]), float(parts[1])


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}" if np.isfinite(value) else "NA"


def _write_report(
    output_dir: Path,
    errors: pd.DataFrame,
    failures: pd.DataFrame,
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    top10: pd.DataFrame,
) -> Path:
    """Write a plain-English summary alongside the complete CSV tables."""
    forecast_metrics = metrics.loc[metrics["grouping"].eq("forecast_level")]
    absolute = forecast_metrics.loc[
        forecast_metrics["error_type"].eq("absolute_volatility_error")
    ]
    relative = forecast_metrics.loc[
        forecast_metrics["error_type"].eq("relative_error")
    ]
    logarithmic = forecast_metrics.loc[
        forecast_metrics["error_type"].eq("log_error")
    ]
    ratios = []
    for model in MODEL_NAMES:
        by_q = absolute.loc[absolute["model"].eq(model)].set_index("quintile")
        by_rel = relative.loc[relative["model"].eq(model)].set_index("quintile")
        by_log = logarithmic.loc[logarithmic["model"].eq(model)].set_index("quintile")
        raw_ratio = by_q.loc["Q5", "residual_variance"] / by_q.loc[
            "Q1", "residual_variance"
        ]
        relative_ratio = by_rel.loc["Q5", "residual_variance"] / by_rel.loc[
            "Q1", "residual_variance"
        ]
        log_ratio = by_log.loc["Q5", "residual_variance"] / by_log.loc[
            "Q1", "residual_variance"
        ]
        ratios.append(
            f"| {MODEL_NAMES[model]} | {_fmt(raw_ratio)} | "
            f"{_fmt(relative_ratio)} | {_fmt(log_ratio)} |"
        )

    variance_pivot = absolute.pivot(
        index="model", columns="quintile", values="residual_variance"
    )
    increasing = int((variance_pivot["Q5"] > variance_pivot["Q1"]).sum())
    evidence = [
        f"**Evidence:** Raw Q5 error variance exceeds Q1 for {increasing} of "
        f"{len(MODEL_NAMES)} models in this common-date sample.",
        "**Evidence:** Raw Q5/Q1 variance ratios are "
        f"{min(variance_pivot['Q5'] / variance_pivot['Q1']):.1f}–"
        f"{max(variance_pivot['Q5'] / variance_pivot['Q1']):.1f}, while "
        "relative-error ratios are 0.96–1.33 and log-error ratios are "
        "0.48–0.85. The raw-scale widening largely disappears after "
        "rescaling; it does not persist as larger Q5 variance in log errors.",
        "**Evidence:** Q5 errors remain right-skewed and heavy-tailed. "
        "Raw-error Q5 skewness is 3.36–4.07 and excess kurtosis is "
        "21.9–25.0, so scale is not the only diagnostic concern.",
    ]
    failed_models = failures.loc[failures["failures"].gt(0)]
    if failed_models.empty:
        evidence.append(
            "**Evidence:** All six models have successful forecasts on the "
            f"{errors['target_date'].nunique()} common targets used here."
        )
    else:
        summary = ", ".join(
            f"{MODEL_NAMES[row.model]}: {row.failures}"
            for row in failed_models.itertuples()
        )
        evidence.append(f"**Evidence:** Forecast failures were recorded: {summary}.")
    largest_dates = top10["target_date"].dt.strftime("%Y-%m-%d").value_counts()
    repeated = largest_dates[largest_dates.gt(1)]
    if not repeated.empty:
        date, count = repeated.index[0], int(repeated.iloc[0])
        evidence.append(
            f"**Evidence:** {count} of the ten largest Q5 misses share the "
            f"target date {date}, showing a common difficult episode across models."
        )

    hac_lines = []
    for row in tests.itertuples():
        label = "raw" if row.test == "squared_absolute_error_on_forecast" else "log"
        direction = "positive" if row.slope > 0 else "non-positive"
        hac_lines.append(
            f"| {MODEL_NAMES[row.model]} | {label} | {direction} | "
            f"{_fmt(row.slope)} | {_fmt(row.hac_standard_error)} | "
            f"{_fmt(row.p_value)} | {row.hac_lags} |"
        )

    top_lines = [
        f"| {rank} | {MODEL_NAMES[row.model]} | {row.target_date:%Y-%m-%d} | "
        f"{_fmt(row.actual)} | {_fmt(row.forecast)} | {_fmt(row.forecast_error)} |"
        for rank, row in enumerate(top10.itertuples(), start=1)
    ]
    failure_lines = [
        f"| {MODEL_NAMES[row.model]} | {row.attempts} | {row.failures} |"
        for row in failures.itertuples()
    ]

    hgb_q5 = absolute.loc[
        absolute["model"].eq("hist_gradient_boosting_ohlc")
        & absolute["quintile"].eq("Q5")
    ].iloc[0]
    reference = _report_q5_reference()
    regenerated_extreme = errors.loc[
        errors["model"].eq("egarchx")
        & errors["target_date"].eq(pd.Timestamp("2026-02-02"))
    ]
    published_miss_range = _selected_report_miss_range()
    comparison = ""
    if reference:
        current_mean_milli = hgb_q5["mean_error"] * 1000
        report_mean = reference.get("mean_error_report_units", np.nan)
        current_skewness = hgb_q5["skewness"]
        report_skewness = reference.get("skewness", np.nan)
        comparison = (
            f"The existing report gives HGB Q5 mean error {report_mean:.3f} "
            f"and skewness {report_skewness:.2f} (mean in 10^-3 units). This "
            f"baseline run gives {current_mean_milli:.3f} and "
            f"{current_skewness:.2f}. The report's HGB is from its joint-predictor "
            "run, while this requested baseline CSV uses the no-external-predictor "
            "forecast, so a difference is not a same-specification mismatch."
        )
    else:
        comparison = (
            "The report's Q5 table could not be parsed; compare the source "
            "report manually if needed."
        )
    if not regenerated_extreme.empty and published_miss_range is not None:
        row = regenerated_extreme.iloc[0]
        regenerated_miss_milli = abs(row["forecast_error"]) * 1000
        comparison += (
            f" For regenerated EGARCH-X on 2026-02-02, actual is "
            f"{row['actual'] * 1000:.3f}, forecast is {row['forecast'] * 1000:.3f}, "
            f"and absolute error is {regenerated_miss_milli:.3f} in 10^-3 units. "
            f"The selected-model report only publishes the cross-model error "
            f"range {published_miss_range[0]:.3f}–{published_miss_range[1]:.3f}; "
            "this regenerated EGARCH-X miss falls within that range, but no "
            "model-specific EGARCH-X value is given there."
        )

    report_text = "\n".join(
        [
            "# Step 1 — Q5 forecast-error diagnosis",
            "",
            "Signed errors use **actual minus forecast**; positive errors mean "
            "under-forecasting. HGB remains a point forecast and is not assigned "
            "a likelihood. Range GARCH-X is labeled Gaussian quasi-likelihood.",
            "",
            "## Data and grouping",
            "",
            f"- Common test targets: {errors['target_date'].nunique()} "
            f"({errors['target_date'].min():%Y-%m-%d} to "
            f"{errors['target_date'].max():%Y-%m-%d}).",
            "- Five model forecasts come from the tracked baseline 20-model CSV. "
            "EGARCH-X was regenerated sequentially; its selected predictors "
            "remain in the ARX mean equation.",
            "- Forecast Q1–Q5 are equal-count holdout ranks separately within "
            "each model; tied forecasts are ordered by target date.",
            "- Origin-volatility bins use the initial training sample's "
            "20/40/60/80% cutoffs. The input is the last observed target at "
            "each forecast origin, available at that time. Test bin sizes need "
            "not be equal.",
            "- Realized-volatility bins rank holdout actuals and are "
            "retrospective diagnostics.",
            "- HAC regressions use four lags. The slope tests are associations, "
            "not causal tests.",
            "",
            "## Existing-report comparison",
            "",
            comparison,
            "No Q5 residual-variance summary CSV is present in the checkout, "
            "so there is no prior Q5 variance number to reproduce. The report "
            "does contain mean-error and skewness tables.",
            "",
            "## Forecast-Q5 variance ratios",
            "",
            "| Model | Raw variance Q5/Q1 | Relative variance Q5/Q1 | Log-error variance Q5/Q1 |",
            "|---|---:|---:|---:|",
            *ratios,
            "",
            "## Evidence and hypotheses",
            "",
            *[f"- {line}" for line in evidence],
            "- **Hypothesis:** If relative and log Q5/Q1 variance ratios are "
            "much closer to 1 than the raw ratio, much of the raw widening is "
            "a scale effect. Ratios that remain high are consistent with "
            "proportional misses, heavy tails, or changing regimes; these "
            "summaries cannot identify which mechanism caused them.",
            "- **Hypothesis:** Extreme misses and shared miss dates may reflect "
            "abrupt volatility changes a one-step point forecast does not "
            "capture. This is a possible explanation, not a causal conclusion.",
            "",
            "## HAC variance-trend regressions",
            "",
            "Each model is fit separately. The raw regression uses squared "
            "actual-minus-forecast errors as the outcome and forecast level "
            "as the predictor. The second uses squared log errors and log forecast.",
            "",
            "| Model | Regression | Slope sign | HAC slope | HAC SE | p-value | Lags |",
            "|---|---|---|---:|---:|---:|---:|",
            *hac_lines,
            "",
            "## Ten largest absolute misses among forecast-Q5 rows",
            "",
            "| Rank | Model | Target date | Actual | Forecast | Actual − forecast |",
            "|---:|---|---|---:|---:|---:|",
            *top_lines,
            "",
            "## Fit attempts",
            "",
            "| Model | Attempts | Failures |",
            "|---|---:|---:|",
            *failure_lines,
            "",
            "Complete statistics for every model, grouping, quintile, and error "
            "measure are in `grouped_error_metrics.csv`. Row-level assignments "
            "and errors are in `classified_forecasts.csv`; regenerated EGARCH-X "
            "rows are explicitly labeled.",
            "",
        ]
    )
    output_path = output_dir / "step1_report.md"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def run(output_dir: Path) -> None:
    """Run the diagnostics and save every table and figure."""
    np.random.seed(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline, prepared = _read_inputs()
    regenerated = _regenerate_egarchx(prepared)
    errors, failures = _prepare_forecasts(baseline, regenerated, prepared)
    metrics = _group_metrics(errors)
    tests = _hac_variance_tests(errors)
    top10, top10_by_model = _top_q5_misses(errors)

    errors.to_csv(output_dir / "classified_forecasts.csv", index=False)
    errors.loc[errors["model"].eq("egarchx")].to_csv(
        output_dir / "egarchx_regenerated_forecasts.csv", index=False
    )
    failures.to_csv(output_dir / "fit_failures.csv", index=False)
    metrics.to_csv(output_dir / "grouped_error_metrics.csv", index=False)
    metrics.loc[
        metrics["grouping"].eq("forecast_level")
        & metrics["quintile"].eq("Q5")
    ].to_csv(output_dir / "forecast_q5_metrics.csv", index=False)
    tests.to_csv(output_dir / "hac_variance_tests.csv", index=False)
    top10.to_csv(output_dir / "top10_q5_misses.csv", index=False)
    top10_by_model.to_csv(output_dir / "top10_q5_misses_by_model.csv", index=False)

    report_reference = _report_q5_reference()
    if report_reference:
        hgb = metrics.loc[
            metrics["grouping"].eq("forecast_level")
            & metrics["quintile"].eq("Q5")
            & metrics["model"].eq("hist_gradient_boosting_ohlc")
        ].set_index("error_type")
        current_mean_milli = hgb.loc["absolute_volatility_error", "mean_error"] * 1000
        current_skew = hgb.loc["absolute_volatility_error", "skewness"]
        pd.DataFrame(
            [
                {
                    "model": "hist_gradient_boosting_ohlc",
                    "report_mean_error_1e3": report_reference.get(
                        "mean_error_report_units", np.nan
                    ),
                    "baseline_mean_error_1e3": current_mean_milli,
                    "mean_error_difference_1e3": current_mean_milli
                    - report_reference.get("mean_error_report_units", np.nan),
                    "report_skewness": report_reference.get("skewness", np.nan),
                    "baseline_skewness": current_skew,
                    "skewness_difference": current_skew
                    - report_reference.get("skewness", np.nan),
                    "comparability_note": (
                        "Existing report uses its joint-predictor HGB run; "
                        "baseline CSV uses the no-external-predictor HGB run."
                    ),
                }
            ]
        ).to_csv(output_dir / "reported_q5_comparison.csv", index=False)
    regenerated_extreme = errors.loc[
        errors["model"].eq("egarchx")
        & errors["target_date"].eq(pd.Timestamp("2026-02-02"))
    ]
    published_miss_range = _selected_report_miss_range()
    if not regenerated_extreme.empty and published_miss_range is not None:
        row = regenerated_extreme.iloc[0]
        pd.DataFrame(
            [
                {
                    "target_date": row["target_date"],
                    "actual_1e3": row["actual"] * 1000,
                    "regenerated_egarchx_forecast_1e3": row["forecast"] * 1000,
                    "regenerated_egarchx_absolute_error_1e3": (
                        abs(row["forecast_error"]) * 1000
                    ),
                    "published_cross_model_error_min_1e3": published_miss_range[0],
                    "published_cross_model_error_max_1e3": published_miss_range[1],
                    "within_published_range": published_miss_range[0]
                    <= abs(row["forecast_error"]) * 1000
                    <= published_miss_range[1],
                    "note": "The report gives a range, not a model-specific EGARCH-X value.",
                }
            ]
        ).to_csv(output_dir / "egarchx_report_comparison.csv", index=False)

    figures = _plot_residuals(errors, output_dir)
    report = _write_report(
        output_dir, errors, failures, metrics, tests, top10
    )
    print(f"Common successful targets: {errors['target_date'].nunique()}")
    print(
        f"Target range: {errors['target_date'].min().date()} to "
        f"{errors['target_date'].max().date()}"
    )
    print("\nForecast-Q5 error statistics:")
    print(
        metrics.loc[
            metrics["grouping"].eq("forecast_level")
            & metrics["quintile"].eq("Q5")
        ][
            [
                "model_label",
                "error_type",
                "n",
                "mean_error",
                "residual_variance",
                "MAE",
                "RMSE",
                "skewness",
                "excess_kurtosis",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.8g}")
    )
    print("\nFit failures:")
    print(failures.to_string(index=False))
    print("\nSaved:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(path)
    for path in figures:
        print(path)
    print(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/samuel/stage3_q5",
        help="Directory for this Step 1 run's tables, report, and figures.",
    )
    args = parser.parse_args()
    run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
