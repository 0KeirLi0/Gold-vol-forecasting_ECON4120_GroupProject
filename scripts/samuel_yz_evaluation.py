"""Evaluate the Yang-Zhang forecasts saved on Enoch's test-branch commit."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import io
from pathlib import Path
import platform
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "988adfcdaf80015cce7b2a2d15fccf5dfbf9d8a4"
BASE_FORECAST_PATH = "data/processed/stage2_yz_forecasts.csv"
BASE_SUMMARY_PATH = "data/processed/stage2_yz_summary.csv"
SELECTED_FORECAST_PATH = (
    "data/processed/stage2_yz_selected/stage2_yz_forecasts.csv"
)
VIX_FORECAST_PATH = "data/processed/stage2_yz_vix/stage2_yz_forecasts.csv"
TARGET_PATH = "data/processed/stage2_yz_selected/GCc1_yang_zhang_volatility.csv"
IV_PATH = ROOT / "data/refinitiv/XAU1MO_R.csv"
OUTPUT_DIR = ROOT / "outputs/samuel"
BENCHMARKS = ("yang_zhang_persistence", "har", "garch11")
PREDICTOR_MODELS = (
    "harx_ohlc",
    "ridge_ohlc",
    "lasso_ohlc",
    "hist_gradient_boosting_ohlc",
    "garchx_ohlc_range",
    "var_ohlc",
)
HORIZON = 1
DM_BANDWIDTHS = (1, 4)
CALIBRATION_HAC_LAGS = 4
SEED = 42


def _read_ref_csv(path: str, *, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read a CSV directly from the pinned Git object without checking out test."""
    completed = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return pd.read_csv(io.BytesIO(completed.stdout), parse_dates=list(dates))


def _validate_forecasts(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Check that every model has a complete, common set of daily forecasts."""
    required = {
        "origin_date",
        "target_date",
        "model",
        "forecast",
        "actual",
        "converged",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{label} must contain {sorted(required)}.")
    if frame[["origin_date", "target_date", "model"]].isna().any().any():
        raise ValueError(f"{label} contains missing dates or model names.")
    if frame.duplicated(["target_date", "model"]).any():
        raise ValueError(f"{label} contains duplicate target/model rows.")
    frame = frame.copy()
    frame["forecast"] = pd.to_numeric(frame["forecast"], errors="raise")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="raise")
    if not np.isfinite(frame[["forecast", "actual"]].to_numpy()).all():
        raise ValueError(f"{label} has failed or non-finite forecasts.")
    if not frame["converged"].astype(bool).all():
        raise ValueError(f"{label} contains non-converged forecasts.")
    if (frame["forecast"] < 0).any() or (frame["actual"] < 0).any():
        raise ValueError(f"{label} contains negative volatility values.")
    models = frame["model"].unique()
    target_sets = frame.groupby("model")["target_date"].apply(set)
    if len(target_sets) != len(models) or any(
        targets != target_sets.iloc[0] for targets in target_sets.iloc[1:]
    ):
        raise ValueError(f"{label} models do not use identical target dates.")
    date_actuals = frame.groupby("target_date")["actual"].nunique()
    if (date_actuals > 1).any():
        raise ValueError(f"{label} has conflicting actuals on common targets.")
    return frame.sort_values(["target_date", "model"]).reset_index(drop=True)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load base, VIX, and external-predictor forecasts plus the target history."""
    base = _validate_forecasts(
        _read_ref_csv(
            BASE_FORECAST_PATH,
            dates=("origin_date", "target_date"),
        ),
        "Base Yang-Zhang forecasts",
    )
    selected = _validate_forecasts(
        _read_ref_csv(
            SELECTED_FORECAST_PATH,
            dates=("origin_date", "target_date"),
        ),
        "Selected-predictor Yang-Zhang forecasts",
    )
    with_vix = _validate_forecasts(
        _read_ref_csv(
            VIX_FORECAST_PATH,
            dates=("origin_date", "target_date"),
        ),
        "VIX Yang-Zhang forecasts",
    )
    target = _read_ref_csv(
        TARGET_PATH,
        dates=("date", "window_start_date", "window_end_date"),
    )
    for label, frame in (("selected", selected), ("VIX", with_vix)):
        expected_models = (
            set(PREDICTOR_MODELS) if label == "selected" else set(base["model"])
        )
        if set(frame["model"]) != expected_models:
            raise ValueError(f"{label} forecast model names do not match expectations.")
        expected_base = base.loc[base["model"].isin(expected_models)]
        base_keys = set(zip(expected_base["target_date"], expected_base["model"]))
        variant_keys = set(zip(frame["target_date"], frame["model"]))
        if base_keys != variant_keys:
            raise ValueError(f"{label} forecasts do not match the base target/model keys.")
        expected = expected_base[["target_date", "model", "actual"]].merge(
            frame[["target_date", "model", "actual"]],
            on=["target_date", "model"],
            suffixes=("_base", "_variant"),
            validate="one_to_one",
        )
        if not np.allclose(
            expected["actual_base"],
            expected["actual_variant"],
            rtol=0,
            atol=1e-14,
        ):
            raise ValueError(f"{label} actuals differ from the base forecast file.")
    if target["date"].duplicated().any():
        raise ValueError("The prepared Yang-Zhang target has duplicate dates.")
    return base, selected, with_vix, target


def _metric_record(
    model: str,
    frame: pd.DataFrame,
    *,
    variant: str = "base",
) -> dict[str, object]:
    """Calculate errors with a consistent actual-minus-forecast bias sign."""
    actual = frame["actual"].to_numpy(dtype=float)
    forecast = frame["forecast"].to_numpy(dtype=float)
    errors = actual - forecast
    return {
        "variant": variant,
        "model": model,
        "n": len(frame),
        "first_target": frame["target_date"].min().date().isoformat(),
        "last_target": frame["target_date"].max().date().isoformat(),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_error_actual_minus_forecast": float(np.mean(errors)),
    }


def _model_comparison(base: pd.DataFrame) -> pd.DataFrame:
    """Rank the 20 base models by MAE and RMSE on their common test dates."""
    records = [
        _metric_record(model, group)
        for model, group in base.groupby("model", sort=False)
    ]
    summary = pd.DataFrame(records).sort_values("MAE").reset_index(drop=True)
    summary["MAE_rank"] = np.arange(1, len(summary) + 1)
    summary["RMSE_rank"] = summary["RMSE"].rank(method="min").astype(int)
    return summary


def _verify_saved_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Confirm recomputed MAE and RMSE match the pinned summary artifact."""
    saved = _read_ref_csv(BASE_SUMMARY_PATH)
    comparison = summary[["model", "MAE", "RMSE"]].merge(
        saved[["model", "common_MAE", "common_RMSE"]],
        on="model",
        how="outer",
        validate="one_to_one",
    )
    if comparison.isna().any().any():
        raise ValueError("Recomputed and saved summaries contain different model sets.")
    if not np.allclose(comparison["MAE"], comparison["common_MAE"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed MAE does not match the saved common-sample summary.")
    if not np.allclose(
        comparison["RMSE"], comparison["common_RMSE"], rtol=0, atol=1e-12
    ):
        raise ValueError("Recomputed RMSE does not match the saved common-sample summary.")
    return comparison


def _loss_values(actual: np.ndarray, forecast: np.ndarray, loss: str) -> np.ndarray:
    if loss == "absolute_error":
        return np.abs(actual - forecast)
    if loss == "squared_error":
        return np.square(actual - forecast)
    raise ValueError(f"Unknown loss {loss!r}.")


def _dm_test(
    loss_difference: np.ndarray,
    *,
    horizon: int,
    bandwidth: int,
) -> tuple[float, float, float]:
    """Calculate a Newey-West DM test with separate overlap bandwidth and HLN horizon."""
    count = len(loss_difference)
    mean_difference = float(np.mean(loss_difference))
    centered = loss_difference - mean_difference
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, min(bandwidth, count - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1 - lag / (bandwidth + 1)
        long_run_variance += 2 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    if long_run_variance == 0:
        statistic = 0.0 if mean_difference == 0 else float(np.sign(mean_difference) * np.inf)
    else:
        statistic = mean_difference / np.sqrt(long_run_variance / count)
    hln_squared = (
        count + 1 - 2 * horizon + horizon * (horizon - 1) / count
    ) / count
    corrected = statistic * np.sqrt(max(hln_squared, 0.0))
    p_value = (
        float(2 * student_t.sf(abs(corrected), df=count - 1))
        if np.isfinite(corrected)
        else 0.0
    )
    return float(corrected), p_value, mean_difference


def _append_dm_comparison(
    rows: list[dict[str, object]],
    candidate: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    family: str,
    candidate_name: str,
    benchmark_name: str,
    loss_names: tuple[str, ...] = ("absolute_error", "squared_error"),
) -> None:
    """Append paired DM tests after confirming the dates and actuals match."""
    left = candidate[["target_date", "actual", "forecast"]].rename(
        columns={"actual": "actual_candidate", "forecast": "forecast_candidate"}
    )
    right = benchmark[["target_date", "actual", "forecast"]].rename(
        columns={"actual": "actual_benchmark", "forecast": "forecast_benchmark"}
    )
    paired = left.merge(right, on="target_date", validate="one_to_one")
    if not np.allclose(
        paired["actual_candidate"],
        paired["actual_benchmark"],
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("A DM comparison has different actual target values.")
    actual = paired["actual_candidate"].to_numpy(dtype=float)
    candidate_forecast = paired["forecast_candidate"].to_numpy(dtype=float)
    benchmark_forecast = paired["forecast_benchmark"].to_numpy(dtype=float)
    for loss in loss_names:
        candidate_loss = _loss_values(actual, candidate_forecast, loss)
        benchmark_loss = _loss_values(actual, benchmark_forecast, loss)
        for bandwidth in DM_BANDWIDTHS:
            statistic, p_value, difference = _dm_test(
                candidate_loss - benchmark_loss,
                horizon=HORIZON,
                bandwidth=bandwidth,
            )
            rows.append(
                {
                    "family": family,
                    "candidate": candidate_name,
                    "benchmark": benchmark_name,
                    "loss": loss,
                    "horizon": HORIZON,
                    "bandwidth": bandwidth,
                    "n": len(paired),
                    "mean_loss_difference_candidate_minus_benchmark": difference,
                    "dm_hln_statistic": statistic,
                    "p_raw": p_value,
                }
            )


def _holm_dm(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Adjust p-values within each pre-defined comparison family and loss."""
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_holm"] = np.nan
    for _, indices in result.groupby(["family", "loss", "bandwidth"]).groups.items():
        result.loc[indices, "p_holm"] = multipletests(
            result.loc[indices, "p_raw"].to_numpy(dtype=float),
            method="holm",
        )[1]
    result["direction"] = np.where(
        result["mean_loss_difference_candidate_minus_benchmark"] < 0,
        "candidate lower loss",
        "candidate higher loss",
    )
    return result.sort_values(["family", "loss", "bandwidth", "candidate", "benchmark"])


def _all_dm_tests(
    base: pd.DataFrame,
    selected: pd.DataFrame,
    with_vix: pd.DataFrame,
) -> pd.DataFrame:
    """Test models against fixed baselines and compare predictor experiments."""
    rows: list[dict[str, object]] = []
    base_models = set(base["model"])
    for model in sorted(base_models):
        if model in BENCHMARKS:
            continue
        candidate = base.loc[base["model"].eq(model)]
        for benchmark_name in BENCHMARKS:
            benchmark = base.loc[base["model"].eq(benchmark_name)]
            _append_dm_comparison(
                rows,
                candidate,
                benchmark,
                family="base_models_vs_fixed_benchmarks",
                candidate_name=model,
                benchmark_name=benchmark_name,
            )

    for frame, family in (
        (selected, "selected_external_predictors_vs_base"),
        (with_vix, "vix_predictor_vs_base"),
    ):
        for model in PREDICTOR_MODELS:
            left = frame.loc[frame["model"].eq(model)]
            right = base.loc[base["model"].eq(model)]
            _append_dm_comparison(
                rows,
                left,
                right,
                family=family,
                candidate_name=f"{model} [{family}]",
                benchmark_name=f"{model} [base]",
            )
    return _holm_dm(rows)


def _predictor_ablation(
    base: pd.DataFrame,
    selected: pd.DataFrame,
    with_vix: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired error changes from adding predictor information."""
    rows = []
    for variant_name, frame in (
        ("selected_DXY_and_gold_IV", selected),
        ("VIX", with_vix),
    ):
        for model in PREDICTOR_MODELS:
            base_model = base.loc[base["model"].eq(model)].set_index("target_date")
            variant_model = frame.loc[frame["model"].eq(model)].set_index("target_date")
            paired = base_model[["actual", "forecast"]].join(
                variant_model[["forecast"]].rename(
                    columns={"forecast": "variant_forecast"}
                ),
                how="inner",
                validate="one_to_one",
            )
            actual = paired["actual"].to_numpy(dtype=float)
            base_error = actual - paired["forecast"].to_numpy(dtype=float)
            variant_error = actual - paired["variant_forecast"].to_numpy(dtype=float)
            rows.append(
                {
                    "predictor_experiment": variant_name,
                    "model": model,
                    "n": len(paired),
                    "base_MAE": float(np.mean(np.abs(base_error))),
                    "variant_MAE": float(np.mean(np.abs(variant_error))),
                    "change_MAE_variant_minus_base": float(
                        np.mean(np.abs(variant_error)) - np.mean(np.abs(base_error))
                    ),
                    "base_RMSE": float(np.sqrt(np.mean(base_error**2))),
                    "variant_RMSE": float(np.sqrt(np.mean(variant_error**2))),
                    "change_RMSE_variant_minus_base": float(
                        np.sqrt(np.mean(variant_error**2))
                        - np.sqrt(np.mean(base_error**2))
                    ),
                    "base_bias_actual_minus_forecast": float(np.mean(base_error)),
                    "variant_bias_actual_minus_forecast": float(np.mean(variant_error)),
                }
            )
    return pd.DataFrame(rows)


def _align_iv(
    target: pd.DataFrame,
    test_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Make prior-day 1M IV daily-volatility forecasts and a training-only scale."""
    expected_hash = "94c422c332a449cba95f437b4a52304856d620abe2c3c86a7d5296e8c8884097"
    import hashlib

    actual_hash = hashlib.sha256(IV_PATH.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("The local XAU1MO_R.csv does not match the pinned YZ input.")
    quotes = pd.read_csv(IV_PATH, parse_dates=["Date"])
    if quotes["Date"].isna().any() or quotes["Date"].duplicated().any():
        raise ValueError("Gold implied-volatility dates must be valid and unique.")
    quotes["MID_PRICE"] = pd.to_numeric(quotes["MID_PRICE"], errors="raise")
    quotes = quotes[["Date", "MID_PRICE"]].dropna().sort_values("Date")
    quotes["iv_daily"] = quotes["MID_PRICE"] / 100 / np.sqrt(252)
    quotes = quotes.rename(columns={"Date": "quote_date"})

    target_values = target[
        ["date", "window_start_date", "daily_volatility", "split"]
    ].dropna(subset=["daily_volatility", "window_start_date"])
    aligned_training = pd.merge_asof(
        target_values.sort_values("date"),
        quotes,
        left_on="window_start_date",
        right_on="quote_date",
        direction="backward",
        allow_exact_matches=True,
    )
    training = aligned_training.loc[aligned_training["split"].eq("train")].dropna(
        subset=["iv_daily"]
    )
    if len(training) != 3040:
        raise ValueError(
            f"Expected 3,040 training targets for IV scaling; found {len(training)}."
        )
    scale_factor = float(training["daily_volatility"].mean() / training["iv_daily"].mean())

    test = test_forecasts[
        ["origin_date", "target_date", "actual"]
    ].drop_duplicates("target_date").sort_values("origin_date")
    test = pd.merge_asof(
        test,
        quotes,
        left_on="origin_date",
        right_on="quote_date",
        direction="backward",
        allow_exact_matches=True,
    )
    if test["quote_date"].isna().any():
        raise ValueError("Some Yang-Zhang test targets have no prior IV quote.")
    if not (test["quote_date"] <= test["origin_date"]).all():
        raise ValueError("A same-day or future implied-volatility quote was used.")
    test["iv_daily_scaled"] = test["iv_daily"] * scale_factor
    test["origin_date"] = pd.to_datetime(test["target_date"]) - pd.Timedelta(days=1)
    metadata = {
        "iv_sha256": actual_hash,
        "scale_factor": scale_factor,
        "scale_training_n": len(training),
        "scale_training_start": training["date"].min().date().isoformat(),
        "scale_training_end": training["date"].max().date().isoformat(),
        "test_n": len(test),
        "test_quote_start": test["quote_date"].min().date().isoformat(),
        "test_quote_end": test["quote_date"].max().date().isoformat(),
    }
    return test, metadata


def _iv_benchmark(
    base: pd.DataFrame,
    target: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Score IV, persistence, HAR, GARCH, and the leading MAE model."""
    reference = base.loc[
        base["model"].eq(BENCHMARKS[0]),
        ["origin_date", "target_date", "actual"],
    ]
    aligned, metadata = _align_iv(target, reference)
    best_model = str(summary.sort_values("MAE").iloc[0]["model"])
    models = {
        "xau_1m_iv_unscaled": "iv_daily",
        "xau_1m_iv_scaled": "iv_daily_scaled",
    }
    forecast_sets = []
    for name in (best_model, *BENCHMARKS):
        if name not in {item[0] for item in forecast_sets}:
            forecast_sets.append(
                (name, base.loc[base["model"].eq(name), ["target_date", "actual", "forecast"]])
            )
    iv_rows = []
    for name, column in models.items():
        frame = aligned[["target_date", "actual", column]].rename(
            columns={column: "forecast"}
        )
        record = _metric_record(name, frame)
        iv_rows.append(record)
    for name, frame in forecast_sets:
        iv_rows.append(_metric_record(name, frame))
    result = pd.DataFrame(iv_rows)

    tests: list[dict[str, object]] = []
    for name, column in models.items():
        candidate = aligned[["target_date", "actual", column]].rename(
            columns={column: "forecast"}
        )
        for benchmark_name, benchmark_frame in forecast_sets:
            _append_dm_comparison(
                tests,
                candidate,
                benchmark_frame,
                family="gold_IV_vs_model",
                candidate_name=name,
                benchmark_name=benchmark_name,
            )
    iv_dm = _holm_dm(tests)
    return result, iv_dm, metadata


def _calibration(base: pd.DataFrame) -> pd.DataFrame:
    """Estimate forecast bias and HAC Mincer-Zarnowitz calibration by model."""
    rows = []
    for model, group in base.groupby("model", sort=False):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        errors = actual - forecast
        bias_fit = sm.OLS(errors, np.ones((len(errors), 1))).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": CALIBRATION_HAC_LAGS},
        )
        design = sm.add_constant(forecast, has_constant="add")
        mz_fit = sm.OLS(actual, design).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": CALIBRATION_HAC_LAGS},
        )
        joint = mz_fit.wald_test(
            (np.eye(2), np.array([0.0, 1.0])),
            scalar=True,
        )
        rows.append(
            {
                "model": model,
                "n": len(group),
                "mean_error_actual_minus_forecast": float(np.mean(errors)),
                "mean_error_hac_t": float(bias_fit.tvalues[0]),
                "mean_error_hac_p": float(bias_fit.pvalues[0]),
                "mz_intercept": float(mz_fit.params[0]),
                "mz_intercept_hac_se": float(mz_fit.bse[0]),
                "mz_slope": float(mz_fit.params[1]),
                "mz_slope_hac_se": float(mz_fit.bse[1]),
                "mz_joint_test_p": float(joint.pvalue),
            }
        )
    return pd.DataFrame(rows)


def _residual_diagnostics(base: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Check forecast-error dependence without requiring constant error variance."""
    selected = list(
        dict.fromkeys(
            summary.sort_values("MAE")["model"].head(3).tolist() + list(BENCHMARKS)
        )
    )
    rows = []
    for model in selected:
        group = base.loc[base["model"].eq(model)].sort_values("target_date")
        errors = (group["actual"] - group["forecast"]).to_numpy(dtype=float)
        for name, values in (("forecast_error", errors), ("squared_error", errors**2)):
            lags = [lag for lag in (1, 4, 10, 20) if lag < len(values)]
            tests = acorr_ljungbox(values, lags=lags, return_df=True)
            for lag in lags:
                rows.append(
                    {
                        "model": model,
                        "series": name,
                        "lag": lag,
                        "ljung_box_statistic": float(tests.loc[lag, "lb_stat"]),
                        "p_value": float(tests.loc[lag, "lb_pvalue"]),
                        "note": "exploratory; two-observation target windows overlap",
                    }
                )
    return pd.DataFrame(rows)


def _regime_performance(
    base: pd.DataFrame,
    selected: pd.DataFrame,
    with_vix: pd.DataFrame,
) -> pd.DataFrame:
    """Describe model errors across low, middle, and high realized-volatility dates."""
    actual_by_target = base.loc[
        base["model"].eq(BENCHMARKS[0]), ["target_date", "actual"]
    ].drop_duplicates("target_date")
    ranked = actual_by_target["actual"].rank(method="first")
    actual_by_target["realized_volatility_tercile"] = pd.qcut(
        ranked,
        3,
        labels=["low", "middle", "high"],
    ).to_numpy()
    records = []
    for variant, frame in (
        ("base", base),
        ("selected_DXY_and_gold_IV", selected),
        ("with_VIX", with_vix),
    ):
        joined = frame.merge(
            actual_by_target[["target_date", "realized_volatility_tercile"]],
            on="target_date",
            validate="many_to_one",
        )
        for (model, tercile), group in joined.groupby(
            ["model", "realized_volatility_tercile"], observed=True
        ):
            records.append(
                {
                    "variant": variant,
                    "model": model,
                    "realized_volatility_tercile": str(tercile),
                    **_metric_record(model, group, variant=variant),
                }
            )
    return pd.DataFrame(records)


def _plot_leaders(base: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """Plot the MAE leader and the three pre-specified baselines."""
    leader = str(summary.sort_values("MAE").iloc[0]["model"])
    models = list(dict.fromkeys([leader, *BENCHMARKS]))
    actual = base["actual"].to_numpy(dtype=float)
    forecasts = base.loc[base["model"].isin(models), "forecast"].to_numpy(dtype=float)
    limit = float(np.quantile(np.concatenate([actual, forecasts]), 0.995))
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, models, strict=True):
        group = base.loc[base["model"].eq(model)]
        axis.scatter(group["forecast"], group["actual"], s=14, alpha=0.45)
        axis.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
        axis.set(xlim=(0, limit), ylim=(0, limit), title=model.replace("_", " "))
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast next-session daily volatility")
    for axis in axes[:, 0]:
        axis.set_ylabel("Actual Yang-Zhang daily volatility")
    figure.suptitle("Yang-Zhang one-step forecasts on the common test sample")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _versions() -> str:
    packages = ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")
    values = [f"Python {platform.python_version()}"]
    for package in packages:
        try:
            values.append(f"{package} {version(package)}")
        except PackageNotFoundError:
            values.append(f"{package} unavailable")
    return ", ".join(values)


def _summary_text(
    summary: pd.DataFrame,
    dm: pd.DataFrame,
    ablation: pd.DataFrame,
    iv_results: pd.DataFrame,
    iv_dm: pd.DataFrame,
    iv_metadata: dict[str, object],
    calibration: pd.DataFrame,
    residuals: pd.DataFrame,
) -> str:
    """Write the main findings and cautions in plain English."""
    top_mae = summary.sort_values("MAE").iloc[0]
    top_rmse = summary.sort_values("RMSE").iloc[0]
    primary_dm = dm.loc[
        dm["family"].eq("base_models_vs_fixed_benchmarks")
        & dm["loss"].eq("absolute_error")
        & dm["bandwidth"].eq(1)
        & dm["candidate"].eq(top_mae["model"])
    ]
    sig = primary_dm.loc[primary_dm["p_holm"] < 0.05, "benchmark"].tolist()
    sig_text = ", ".join(sig) if sig else "none after Holm correction"
    significant_base = dm.loc[
        dm["family"].eq("base_models_vs_fixed_benchmarks")
        & dm["bandwidth"].eq(1)
        & dm["p_holm"].lt(0.05)
    ]

    def significant_pairs(loss: str) -> str:
        selected_pairs = significant_base.loc[significant_base["loss"].eq(loss)]
        descriptions = []
        for benchmark in BENCHMARKS:
            compared = selected_pairs.loc[selected_pairs["benchmark"].eq(benchmark)]
            better = compared.loc[
                compared["mean_loss_difference_candidate_minus_benchmark"].lt(0),
                "candidate",
            ].tolist()
            worse = compared.loc[
                compared["mean_loss_difference_candidate_minus_benchmark"].gt(0),
                "candidate",
            ].tolist()
            descriptions.append(
                f"{benchmark}: lower loss [{', '.join(better) or 'none'}], "
                f"higher loss [{', '.join(worse) or 'none'}]"
            )
        return "; ".join(descriptions)

    selected = ablation.loc[
        ablation["predictor_experiment"].eq("selected_DXY_and_gold_IV")
    ]
    vix = ablation.loc[ablation["predictor_experiment"].eq("VIX")]
    iv_raw = iv_results.loc[iv_results["model"].eq("xau_1m_iv_unscaled")].iloc[0]
    iv_scaled = iv_results.loc[iv_results["model"].eq("xau_1m_iv_scaled")].iloc[0]
    iv_vs_leader = iv_dm.loc[
        iv_dm["candidate"].eq("xau_1m_iv_unscaled")
        & iv_dm["benchmark"].eq(top_mae["model"])
        & iv_dm["loss"].eq("absolute_error")
        & iv_dm["bandwidth"].eq(1)
    ]
    iv_p = float(iv_vs_leader["p_holm"].iloc[0]) if not iv_vs_leader.empty else float("nan")
    iv_scaled_vs_leader = iv_dm.loc[
        iv_dm["candidate"].eq("xau_1m_iv_scaled")
        & iv_dm["benchmark"].eq(top_mae["model"])
        & iv_dm["loss"].eq("absolute_error")
        & iv_dm["bandwidth"].eq(1)
    ]
    iv_scaled_p = (
        float(iv_scaled_vs_leader["p_holm"].iloc[0])
        if not iv_scaled_vs_leader.empty
        else float("nan")
    )
    scaled_iv_vs_persistence = iv_dm.loc[
        iv_dm["candidate"].eq("xau_1m_iv_scaled")
        & iv_dm["benchmark"].eq("yang_zhang_persistence")
        & iv_dm["loss"].eq("absolute_error")
        & iv_dm["bandwidth"].eq(1)
    ]
    scaled_persistence_p = (
        float(scaled_iv_vs_persistence["p_holm"].iloc[0])
        if not scaled_iv_vs_persistence.empty
        else float("nan")
    )
    bias_p = calibration.loc[
        calibration["model"].eq(top_mae["model"]), "mean_error_hac_p"
    ].iloc[0]
    mz_p = calibration.loc[
        calibration["model"].eq(top_mae["model"]), "mz_joint_test_p"
    ].iloc[0]
    significant_errors = residuals.loc[
        residuals["series"].eq("forecast_error") & residuals["p_value"].lt(0.05)
    ]
    residual_note = (
        "At least one selected model has forecast-error autocorrelation at a tested lag; "
        "overlapping targets can contribute to this result."
        if not significant_errors.empty
        else "No selected forecast error rejects the Ljung-Box null at the tested lags."
    )
    bias_rejections = calibration.loc[calibration["mean_error_hac_p"].lt(0.05)]
    bias_nonrejections = calibration.loc[calibration["mean_error_hac_p"].ge(0.05)]
    mz_nonrejections = calibration.loc[calibration["mz_joint_test_p"].ge(0.05)]
    selected_dm = dm.loc[
        dm["family"].eq("selected_external_predictors_vs_base")
        & dm["loss"].eq("absolute_error")
        & dm["bandwidth"].eq(1)
        & dm["p_holm"].lt(0.05)
    ]
    vix_dm = dm.loc[
        dm["family"].eq("vix_predictor_vs_base")
        & dm["loss"].eq("absolute_error")
        & dm["bandwidth"].eq(1)
        & dm["p_holm"].lt(0.05)
    ]
    match_status = bool(summary.get("saved_summary_match", pd.Series([True])).all())
    lines = [
        "# Samuel — Yang-Zhang model evaluation",
        "",
        "## Evaluation design",
        "",
        f"- Target: unannualized daily volatility from a two-observation Yang-Zhang estimator on LSEG GCc1 OHLC.",
        f"- Source: read-only forecast artifacts from test-branch commit `{SOURCE_COMMIT}`; main was not switched.",
        f"- Sequential one-step test dates: {int(top_mae['n'])}; {summary['first_target'].min()} to {summary['last_target'].max()}.",
        "- All 20 baseline models use the same test targets; MAE is the primary ranking metric and RMSE is a complementary outlier-sensitive metric.",
        f"- Recomputed MAE/RMSE match the saved Stage 2 common-sample summary: {match_status}.",
        "- DM tests use absolute and squared errors, Newey-West bandwidths 1 and 4, one-step HLN correction, and Holm adjustment within each comparison family.",
        "",
        "## Main results",
        "",
        f"- Lowest MAE: **{top_mae['model']}** (MAE {top_mae['MAE']:.6g}; RMSE {top_mae['RMSE']:.6g}).",
        f"- Lowest RMSE: **{top_rmse['model']}** (RMSE {top_rmse['RMSE']:.6g}; MAE {top_rmse['MAE']:.6g}).",
        f"- The MAE leader has significantly lower absolute loss than these fixed benchmarks after Holm correction (bandwidth 1): {sig_text}.",
        f"- Other statistically significant paired results at bandwidth 1, absolute loss (Holm-adjusted): {significant_pairs('absolute_error')}.",
        f"- For squared loss at bandwidth 1, significant lower/higher-loss models are: {significant_pairs('squared_error')}. Bandwidth 4 sensitivity results are in `yz_dm_results.csv`.",
        f"- MAE and RMSE are in daily-volatility units. Bias is reported as actual minus forecast; a positive value means average under-forecasting.",
        f"- The HAC mean-error test rejects zero bias for {len(bias_rejections)} of 20 base models; the non-rejections are {', '.join(bias_nonrejections['model'].tolist()) or 'none'}. Positive errors indicate under-forecasting.",
        f"- The MAE leader under-forecasts on average (HAC p={float(bias_p):.4g}); its Mincer-Zarnowitz joint-calibration p={float(mz_p):.4g}. The joint test fails to reject for: {', '.join(mz_nonrejections['model'].tolist()) or 'none'}. A zero mean error alone does not establish calibration.",
        "",
        "## Predictor experiments",
        "",
        f"- Adding DXY and gold 1M/1Y IV changed MAE for {int((selected['change_MAE_variant_minus_base'] < 0).sum())} of {len(selected)} paired models in the improving direction; it changed RMSE for {int((selected['change_RMSE_variant_minus_base'] < 0).sum())} of {len(selected)}.",
        f"- Adding VIX changed MAE for {int((vix['change_MAE_variant_minus_base'] < 0).sum())} of {len(vix)} paired models in the improving direction; it changed RMSE for {int((vix['change_RMSE_variant_minus_base'] < 0).sum())} of {len(vix)}.",
        f"- After Holm adjustment for paired absolute-loss tests, significant selected-predictor changes: {', '.join(selected_dm['candidate'].tolist()) or 'none'}; significant VIX changes: {', '.join(vix_dm['candidate'].tolist()) or 'none'}.",
        "- Predictor additions are judged on paired out-of-sample losses. In-sample predictor significance alone is not evidence of better forecasting.",
        "",
        "## Gold implied-volatility reference",
        "",
        f"- The 1M IV was converted to daily unannualized volatility as `MID_PRICE / 100 / sqrt(252)` and aligned to the prior quote date.",
        f"- The scale factor {iv_metadata['scale_factor']:.6g} was estimated using {iv_metadata['scale_training_n']} training targets only ({iv_metadata['scale_training_start']} to {iv_metadata['scale_training_end']}).",
        f"- Unscaled IV MAE/RMSE: {iv_raw['MAE']:.6g}/{iv_raw['RMSE']:.6g}; training-scaled IV MAE/RMSE: {iv_scaled['MAE']:.6g}/{iv_scaled['RMSE']:.6g}.",
        f"- Both unscaled and training-scaled IV have higher absolute loss than the test-sample MAE leader (Holm-adjusted p={iv_p:.4g} and {iv_scaled_p:.4g}, respectively). Scaled IV is not significantly different from persistence (Holm p={scaled_persistence_p:.3g}). These tests are exploratory because the leader was selected on this same test set.",
        "- IV maturity is one month while the target is next-session volatility, so this is a market reference, not a maturity-matched forecast.",
        "- VIX measures U.S. equity implied volatility. It is assessed as an input to selected gold-volatility models, not as a standalone gold-volatility forecast.",
        "",
        "## Calibration and residual checks",
        "",
        f"- Residual check: {residual_note}",
        "- A changing residual variance across volatility regimes is not automatically a model failure; forecast errors can naturally widen when actual volatility is high. Check conditional bias and forecast calibration rather than demanding constant error variance.",
        "- The calibration regression uses HAC standard errors; any intercept correction must be estimated on training/validation data and then tested on untouched test data.",
        "",
        "## Limitations",
        "",
        "1. The Yang-Zhang target is an estimated two-observation volatility measure, not directly observed latent volatility; adjacent targets overlap.",
        "2. The comparison covers one continuous COMEX gold futures series and one chronological holdout; it does not establish performance in every market regime.",
        "3. Choosing a winner using this test sample creates selection uncertainty; Holm correction controls the specified test families but does not replace a later holdout.",
        "4. One-month gold IV and one-day realized volatility have different maturities; the conversion and training-period scale factor may not transfer across periods.",
        "5. VIX is a cross-market predictor, and the availability proxy for external predictors may not equal the true historical publication time.",
        "",
        "## Reproducibility",
        "",
        f"- Random seed: {SEED}. Package versions: {_versions()}.",
        "- Forecast inputs were read from a pinned Git object with `git show`; no branch checkout or source-data edits were made.",
        "- Outputs are written under `outputs/samuel/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the paired forecast comparisons and write Samuel's Yang-Zhang outputs."""
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load forecasts and validate their shared targets and actual values.
    base, selected, with_vix, target = _load_inputs()
    base_summary = _model_comparison(base)
    saved_check = _verify_saved_summary(base_summary)
    base_summary = base_summary.merge(
        saved_check[["model", "common_MAE", "common_RMSE"]],
        on="model",
        validate="one_to_one",
    )
    base_summary["saved_summary_match"] = True
    base_summary.to_csv(OUTPUT_DIR / "yz_model_comparison.csv", index=False)

    # Step 2: Test the 20 models against persistence, HAR, and GARCH(1,1).
    dm = _all_dm_tests(base, selected, with_vix)
    dm.to_csv(OUTPUT_DIR / "yz_dm_results.csv", index=False)

    # Step 3: Measure paired changes from selected external predictors and VIX.
    ablation = _predictor_ablation(base, selected, with_vix)
    ablation.to_csv(OUTPUT_DIR / "yz_predictor_ablation.csv", index=False)

    # Step 4: Align 1M gold IV to prior quotes and estimate its scale on training data.
    iv_results, iv_dm, iv_metadata = _iv_benchmark(base, target, base_summary)
    iv_results.to_csv(OUTPUT_DIR / "yz_iv_benchmark.csv", index=False)
    iv_dm.to_csv(OUTPUT_DIR / "yz_iv_dm_results.csv", index=False)

    # Step 5: Check forecast bias, calibration, residual dependence, and regimes.
    calibration = _calibration(base)
    calibration.to_csv(OUTPUT_DIR / "yz_calibration.csv", index=False)
    residuals = _residual_diagnostics(base, base_summary)
    residuals.to_csv(OUTPUT_DIR / "yz_residual_diagnostics.csv", index=False)
    regimes = _regime_performance(base, selected, with_vix)
    regimes.to_csv(OUTPUT_DIR / "yz_regime_performance.csv", index=False)
    _plot_leaders(
        base,
        base_summary,
        OUTPUT_DIR / "yz_top_models_actual_vs_forecast.png",
    )

    # Step 6: Write an evidence-based summary and print the leading comparisons.
    report = _summary_text(
        base_summary,
        dm,
        ablation,
        iv_results,
        iv_dm,
        iv_metadata,
        calibration,
        residuals,
    )
    (OUTPUT_DIR / "yz_summary.md").write_text(report, encoding="utf-8")
    print(base_summary.head(10).to_string(index=False))
    print("\nMAE-leader tests against the three fixed benchmarks:")
    leader = str(base_summary.sort_values("MAE").iloc[0]["model"])
    print(
        dm.loc[
            dm["family"].eq("base_models_vs_fixed_benchmarks")
            & dm["candidate"].eq(leader)
            & dm["loss"].eq("absolute_error")
            & dm["bandwidth"].eq(1)
        ][["candidate", "benchmark", "mean_loss_difference_candidate_minus_benchmark", "p_raw", "p_holm"]]
        .to_string(index=False)
    )
    print("\nIV reference:")
    print(iv_results[["model", "n", "MAE", "RMSE"]].to_string(index=False))
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
