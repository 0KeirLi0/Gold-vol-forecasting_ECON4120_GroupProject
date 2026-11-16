"""Compare saved Stage 3 gold-volatility forecasts and create Samuel's outputs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

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
FORECAST_PATH = ROOT / "data/processed/stage3_integrated_common.csv"
PANEL_PATH = ROOT / "data/processed/stage2_multivariate.csv"
IV_PATH = ROOT / "data/refinitiv/XAU1MO_R.csv"
OUTPUT_DIR = ROOT / "outputs/samuel"
BENCHMARK = "historical_variance"
HORIZON = 5
DM_BANDWIDTHS = (1, 4)
HAC_LAGS = HORIZON - 1
SEED = 42


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read inputs and check dates, actuals, and common forecast coverage."""
    required_forecasts = {"origin_date", "model", "forecast", "actual"}
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["origin_date"])
    panel = pd.read_csv(
        PANEL_PATH,
        parse_dates=["date", "target_end_date"],
    )
    implied_vol = pd.read_csv(IV_PATH, parse_dates=["Date"])

    if not required_forecasts.issubset(forecasts.columns):
        raise ValueError(f"{FORECAST_PATH} must contain {sorted(required_forecasts)}.")
    if not {"date", "target_end_date", "realized_variance"}.issubset(panel.columns):
        raise ValueError(f"{PANEL_PATH} is missing target columns.")
    if not {"Date", "MID_PRICE"}.issubset(implied_vol.columns):
        raise ValueError(f"{IV_PATH} must contain Date and MID_PRICE.")
    if forecasts["origin_date"].isna().any() or forecasts.duplicated(
        ["origin_date", "model"]
    ).any():
        raise ValueError("Forecast origins must be nonmissing and unique per model.")
    if panel["date"].isna().any() or panel["date"].duplicated().any():
        raise ValueError("Panel dates must be nonmissing and unique.")
    if implied_vol["Date"].isna().any() or implied_vol["Date"].duplicated().any():
        raise ValueError("Implied-volatility dates must be nonmissing and unique.")

    forecasts["forecast"] = pd.to_numeric(forecasts["forecast"], errors="raise")
    forecasts["actual"] = pd.to_numeric(forecasts["actual"], errors="raise")
    panel["realized_variance"] = pd.to_numeric(
        panel["realized_variance"], errors="raise"
    )
    implied_vol["MID_PRICE"] = pd.to_numeric(implied_vol["MID_PRICE"], errors="raise")
    if not np.isfinite(forecasts[["forecast", "actual"]].to_numpy()).all():
        raise ValueError("The common-origin forecasts contain non-finite values.")
    if (forecasts[["forecast", "actual"]] <= 0).any().any():
        raise ValueError("Variance forecasts and actuals must be strictly positive.")

    actual_by_date = panel.set_index("date")["realized_variance"]
    expected = actual_by_date.reindex(forecasts["origin_date"]).to_numpy(dtype=float)
    if np.isnan(expected).any() or not np.allclose(
        expected,
        forecasts["actual"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("Saved forecast actuals do not match the Stage 2 panel.")

    counts = forecasts.groupby("model")["origin_date"].nunique()
    if counts.nunique() != 1:
        raise ValueError("All models must use the same number of forecast origins.")
    origin_sets = forecasts.groupby("model")["origin_date"].apply(set)
    if any(origins != origin_sets.iloc[0] for origins in origin_sets.iloc[1:]):
        raise ValueError("All models must use identical forecast dates.")
    if BENCHMARK not in set(forecasts["model"]):
        raise ValueError(f"Required benchmark {BENCHMARK!r} is missing.")
    return (
        forecasts.sort_values(["origin_date", "model"]).reset_index(drop=True),
        panel.sort_values("date").reset_index(drop=True),
        implied_vol.sort_values("Date").reset_index(drop=True),
    )


def _qlike(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Return Patton's QLIKE loss for positive variance values."""
    if (actual <= 0).any() or (forecast <= 0).any():
        raise ValueError("QLIKE requires positive actual and forecast variances.")
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1


def _newey_west_dm(
    loss_difference: np.ndarray,
    *,
    horizon: int,
    bandwidth: int,
) -> tuple[float, float, float]:
    """Run a two-sided DM test with Bartlett HAC variance and HLN correction."""
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

    correction_squared = (
        count + 1 - 2 * horizon + horizon * (horizon - 1) / count
    ) / count
    corrected = statistic * np.sqrt(max(correction_squared, 0.0))
    p_value = (
        float(2 * student_t.sf(abs(corrected), df=count - 1))
        if np.isfinite(corrected)
        else 0.0
    )
    return float(corrected), p_value, mean_difference


def _losses(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive losses for every model on the shared dates."""
    records = []
    for model, group in frame.groupby("model", sort=False):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        records.append(
            {
                "model": model,
                "n": len(group),
                "first_origin": group["origin_date"].min().date().isoformat(),
                "last_origin": group["origin_date"].max().date().isoformat(),
                "MAE": float(np.mean(np.abs(actual - forecast))),
                "RMSE": float(np.sqrt(np.mean(np.square(actual - forecast)))),
                "MSE": float(np.mean(np.square(actual - forecast))),
                "QLIKE": float(np.mean(_qlike(actual, forecast))),
                "mean_error_actual_minus_forecast": float(np.mean(actual - forecast)),
            }
        )
    result = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)
    result["RMSE_rank"] = np.arange(1, len(result) + 1)
    result["MAE_rank"] = result["MAE"].rank(method="min").astype(int)
    result["QLIKE_rank"] = result["QLIKE"].rank(method="min").astype(int)
    return result


def _dm_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare every model with historical variance using two loss functions."""
    indexed = {
        model: group.set_index("origin_date").sort_index()
        for model, group in frame.groupby("model")
    }
    benchmark = indexed[BENCHMARK]
    rows = []
    for model, candidate in indexed.items():
        if model == BENCHMARK:
            continue
        joined = candidate[["actual", "forecast"]].join(
            benchmark[["forecast"]].rename(columns={"forecast": "benchmark_forecast"}),
            how="inner",
            validate="one_to_one",
        )
        actual = joined["actual"].to_numpy(dtype=float)
        forecast = joined["forecast"].to_numpy(dtype=float)
        benchmark_forecast = joined["benchmark_forecast"].to_numpy(dtype=float)
        for loss_name, loss_candidate, loss_benchmark in (
            (
                "squared_error",
                np.square(actual - forecast),
                np.square(actual - benchmark_forecast),
            ),
            (
                "QLIKE",
                _qlike(actual, forecast),
                _qlike(actual, benchmark_forecast),
            ),
        ):
            for bandwidth in DM_BANDWIDTHS:
                statistic, p_value, mean_difference = _newey_west_dm(
                    loss_candidate - loss_benchmark,
                    horizon=HORIZON,
                    bandwidth=bandwidth,
                )
                rows.append(
                    {
                        "candidate": model,
                        "benchmark": BENCHMARK,
                        "loss": loss_name,
                        "bandwidth": bandwidth,
                        "n": len(joined),
                        "mean_loss_difference_candidate_minus_benchmark": mean_difference,
                        "dm_hln_statistic": statistic,
                        "p_raw": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    for _, indices in result.groupby(["loss", "bandwidth"]).groups.items():
        result.loc[indices, "p_holm"] = multipletests(
            result.loc[indices, "p_raw"].to_numpy(dtype=float),
            method="holm",
        )[1]
    result["direction"] = np.where(
        result["mean_loss_difference_candidate_minus_benchmark"] < 0,
        "candidate lower loss",
        "candidate higher loss",
    )

    # This extra row checks whether the VIX-GARCH-X model differs from GARCH(1,1).
    vix_model, garch_model = "garchx_vix", "garch11"
    if vix_model in indexed and garch_model in indexed:
        joined = indexed[vix_model][["actual", "forecast"]].join(
            indexed[garch_model][["forecast"]].rename(
                columns={"forecast": "benchmark_forecast"}
            ),
            how="inner",
            validate="one_to_one",
        )
        actual = joined["actual"].to_numpy(dtype=float)
        forecast = joined["forecast"].to_numpy(dtype=float)
        reference = joined["benchmark_forecast"].to_numpy(dtype=float)
        for loss_name, left, right in (
            ("squared_error", np.square(actual - forecast), np.square(actual - reference)),
            ("QLIKE", _qlike(actual, forecast), _qlike(actual, reference)),
        ):
            statistic, p_value, difference = _newey_west_dm(
                left - right,
                horizon=HORIZON,
                bandwidth=HORIZON - 1,
            )
            result = pd.concat(
                [
                    result,
                    pd.DataFrame(
                        [
                            {
                                "candidate": vix_model,
                                "benchmark": garch_model,
                                "loss": loss_name,
                                "bandwidth": HORIZON - 1,
                                "n": len(joined),
                                "mean_loss_difference_candidate_minus_benchmark": difference,
                                "dm_hln_statistic": statistic,
                                "p_raw": p_value,
                                "p_holm": np.nan,
                                "direction": (
                                    "candidate lower loss"
                                    if difference < 0
                                    else "candidate higher loss"
                                ),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return result.sort_values(["benchmark", "loss", "bandwidth", "candidate"])


def _calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate forecast bias and HAC Mincer-Zarnowitz calibration by model."""
    rows = []
    for model, group in frame.groupby("model", sort=False):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        errors = actual - forecast
        bias_fit = sm.OLS(errors, np.ones((len(errors), 1))).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": HAC_LAGS},
        )
        design = sm.add_constant(forecast, has_constant="add")
        mz_fit = sm.OLS(actual, design).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": HAC_LAGS},
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


def _implied_volatility(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame,
    implied_vol: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Convert gold implied volatility to a five-session variance reference."""
    quotes = implied_vol[["Date", "MID_PRICE"]].dropna().sort_values("Date")
    quotes = quotes.rename(columns={"Date": "quote_date"})
    target_data = panel[
        ["date", "target_end_date", "realized_variance"]
    ].rename(
        columns={
            "date": "origin_date",
            "realized_variance": "actual",
        }
    )
    aligned = pd.merge_asof(
        target_data.sort_values("origin_date"),
        quotes,
        left_on="origin_date",
        right_on="quote_date",
        direction="backward",
        allow_exact_matches=False,
    )
    if (aligned["quote_date"] >= aligned["origin_date"]).any():
        raise ValueError("An implied-volatility quote is not strictly prior to origin.")
    annualized_daily_vol = aligned["MID_PRICE"] / 100 / np.sqrt(252)
    aligned["iv_5session_variance"] = HORIZON * np.square(annualized_daily_vol)

    test_start = forecasts["origin_date"].min()
    training = aligned.loc[
        (aligned["origin_date"] < test_start)
        & (aligned["target_end_date"] < test_start)
        & aligned["actual"].gt(0)
        & aligned["iv_5session_variance"].gt(0)
    ].copy()
    if training.empty:
        raise ValueError("No matured pre-test observations are available for IV scaling.")
    scale_factor = float(
        training["actual"].mean() / training["iv_5session_variance"].mean()
    )

    test = forecasts.loc[
        forecasts["model"].eq(BENCHMARK), ["origin_date", "actual"]
    ].merge(
        aligned[["origin_date", "quote_date", "MID_PRICE", "iv_5session_variance"]],
        on="origin_date",
        how="left",
        validate="one_to_one",
    )
    test["iv_5session_variance_scaled"] = (
        test["iv_5session_variance"] * scale_factor
    )
    test = test.dropna(
        subset=["quote_date", "iv_5session_variance", "iv_5session_variance_scaled"]
    )
    if test.empty:
        raise ValueError("No implied-volatility quotes align with evaluation origins.")
    test["quote_below_1pct_flag"] = test["MID_PRICE"] < 1
    test_start_date = test["origin_date"].min()
    test_end_date = test["origin_date"].max()
    candidate_models = [
        ("xau_1m_iv_unscaled", "iv_5session_variance"),
        ("xau_1m_iv_scaled", "iv_5session_variance_scaled"),
    ]
    best_model = str(summary.sort_values("RMSE").iloc[0]["model"])
    test_forecasts = forecasts.loc[
        forecasts["origin_date"].isin(test["origin_date"])
    ].pivot(index="origin_date", columns="model", values="forecast")
    actuals = test.set_index("origin_date")["actual"]
    records = []
    comparison_losses: dict[str, dict[str, np.ndarray]] = {}
    iv_on_dates = test.set_index("origin_date")
    for name, column in candidate_models:
        merged = iv_on_dates.join(test_forecasts, how="inner")
        actual = merged["actual"].to_numpy(dtype=float)
        forecast = merged[column].to_numpy(dtype=float)
        records.append(
            {
                "model": name,
                "n": len(merged),
                "MAE": float(np.mean(np.abs(actual - forecast))),
                "RMSE": float(np.sqrt(np.mean(np.square(actual - forecast)))),
                "MSE": float(np.mean(np.square(actual - forecast))),
                "QLIKE": float(np.mean(_qlike(actual, forecast))),
                "mean_error_actual_minus_forecast": float(np.mean(actual - forecast)),
                "scale_factor_from_pretest_targets": scale_factor,
                "scale_training_n": len(training),
                "scale_training_end_origin": training["origin_date"].max().date().isoformat(),
                "quote_lag_note": "previous available quote; same-day quote excluded",
            }
        )
        comparison_losses[name] = {
            "squared_error": np.square(actual - forecast),
            "QLIKE": _qlike(actual, forecast),
        }
    for name in (best_model, BENCHMARK):
        actual = actuals.loc[iv_on_dates.index].to_numpy(dtype=float)
        forecast = test_forecasts.loc[iv_on_dates.index, name].to_numpy(dtype=float)
        records.append(
            {
                "model": name,
                "n": len(actual),
                "MAE": float(np.mean(np.abs(actual - forecast))),
                "RMSE": float(np.sqrt(np.mean(np.square(actual - forecast)))),
                "MSE": float(np.mean(np.square(actual - forecast))),
                "QLIKE": float(np.mean(_qlike(actual, forecast))),
                "mean_error_actual_minus_forecast": float(np.mean(actual - forecast)),
                "scale_factor_from_pretest_targets": scale_factor,
                "scale_training_n": len(training),
                "scale_training_end_origin": training["origin_date"].max().date().isoformat(),
                "quote_lag_note": "comparison on dates with available prior IV quotes",
            }
        )
    result = pd.DataFrame(records)
    result["quotes_below_1pct_in_test"] = int(test["quote_below_1pct_flag"].sum())
    result["RMSE_sensitivity_excluding_quotes_below_1pct"] = np.nan
    result["QLIKE_sensitivity_excluding_quotes_below_1pct"] = np.nan
    clean = test.loc[~test["quote_below_1pct_flag"]].set_index("origin_date")
    if not clean.empty:
        clean_actual = clean["actual"].to_numpy(dtype=float)
        clean_forecasts = test_forecasts.loc[clean.index]
        for name, column in candidate_models:
            row = result.index[result["model"].eq(name)][0]
            clean_forecast = clean[column].to_numpy(dtype=float)
            result.loc[row, "RMSE_sensitivity_excluding_quotes_below_1pct"] = float(
                np.sqrt(np.mean(np.square(clean_actual - clean_forecast)))
            )
            result.loc[row, "QLIKE_sensitivity_excluding_quotes_below_1pct"] = float(
                np.mean(_qlike(clean_actual, clean_forecast))
            )

    # Report nominal DM tests against the test-sample RMSE winner as exploratory.
    best_actual = actuals.loc[iv_on_dates.index].to_numpy(dtype=float)
    best_forecast = test_forecasts.loc[iv_on_dates.index, best_model].to_numpy(dtype=float)
    p_values = {"squared_error": [], "QLIKE": []}
    for name in (candidate for candidate, _ in candidate_models):
        for loss_name in ("squared_error", "QLIKE"):
            best_loss = (
                np.square(best_actual - best_forecast)
                if loss_name == "squared_error"
                else _qlike(best_actual, best_forecast)
            )
            _, p_value, _ = _newey_west_dm(
                comparison_losses[name][loss_name] - best_loss,
                horizon=HORIZON,
                bandwidth=HORIZON - 1,
            )
            p_values[loss_name].append(p_value)
    result["dm_vs_test_rmse_winner_mse_p_raw"] = np.nan
    result["dm_vs_test_rmse_winner_mse_p_holm"] = np.nan
    result["dm_vs_test_rmse_winner_qlike_p_raw"] = np.nan
    result["dm_vs_test_rmse_winner_qlike_p_holm"] = np.nan
    for i, name in enumerate(candidate for candidate, _ in candidate_models):
        row = result.index[result["model"].eq(name)][0]
        result.loc[row, "dm_vs_test_rmse_winner_mse_p_raw"] = p_values[
            "squared_error"
        ][i]
        result.loc[row, "dm_vs_test_rmse_winner_qlike_p_raw"] = p_values["QLIKE"][i]
    for loss_name, raw_values in p_values.items():
        adjusted = multipletests(raw_values, method="holm")[1]
        column = (
            "dm_vs_test_rmse_winner_mse_p_holm"
            if loss_name == "squared_error"
            else "dm_vs_test_rmse_winner_qlike_p_holm"
        )
        for i, name in enumerate(candidate for candidate, _ in candidate_models):
            row = result.index[result["model"].eq(name)][0]
            result.loc[row, column] = adjusted[i]

    # Keep only test rows used for comparisons and return IV alignment metadata.
    metadata = {
        "test_n_with_prior_quote": len(test),
        "test_first_origin": test_start_date.date().isoformat(),
        "test_last_origin": test_end_date.date().isoformat(),
        "first_test_origin": test_start.date().isoformat(),
        "iv_scale_factor": scale_factor,
        "scale_training_n": len(training),
        "scale_training_end_origin": training["origin_date"].max().date().isoformat(),
        "iv_conversion": "five-session variance = 5 * (MID_PRICE / 100 / sqrt(252))^2",
        "quotes_below_1pct_in_test": int(test["quote_below_1pct_flag"].sum()),
        "low_quote_rows": test.loc[
            test["quote_below_1pct_flag"],
            ["origin_date", "quote_date", "MID_PRICE"],
        ].to_dict("records"),
    }
    return result, metadata, test


def _residual_diagnostics(
    forecasts: pd.DataFrame,
    implied_test: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Check serial dependence in errors for leading models and the baseline."""
    leaders = summary.sort_values("RMSE")["model"].head(3).tolist()
    selected = list(dict.fromkeys(leaders + [BENCHMARK]))
    frames = [
        forecasts.loc[forecasts["model"].isin(selected), ["origin_date", "model", "forecast", "actual"]]
    ]
    if not implied_test.empty:
        for label, column in (
            ("xau_1m_iv_unscaled", "iv_5session_variance"),
            ("xau_1m_iv_scaled", "iv_5session_variance_scaled"),
        ):
            frames.append(
                implied_test[["origin_date", "actual", column]].rename(
                    columns={column: "forecast"}
                ).assign(model=label)
            )
    data = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for model, group in data.groupby("model", sort=False):
        errors = (
            group.sort_values("origin_date")["actual"].to_numpy(dtype=float)
            - group.sort_values("origin_date")["forecast"].to_numpy(dtype=float)
        )
        for series_name, values in (("error", errors), ("squared_error", errors**2)):
            test_lags = [lag for lag in (5, 10, 20) if lag < len(values)]
            test = acorr_ljungbox(values, lags=test_lags, return_df=True)
            for lag in test_lags:
                rows.append(
                    {
                        "model": model,
                        "series": series_name,
                        "lag": lag,
                        "ljung_box_statistic": float(test.loc[lag, "lb_stat"]),
                        "p_value": float(test.loc[lag, "lb_pvalue"]),
                        "interpretation_note": (
                            "exploratory; overlapping five-session targets can induce dependence"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _plot_top_models(
    forecasts: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Plot actual versus forecast for the three leaders and historical baseline."""
    leaders = summary.sort_values("RMSE")["model"].head(3).tolist()
    models = list(dict.fromkeys(leaders + [BENCHMARK]))
    actual = forecasts["actual"].to_numpy(dtype=float)
    forecast_values = forecasts.loc[
        forecasts["model"].isin(models), "forecast"
    ].to_numpy(dtype=float)
    limit = float(np.quantile(np.concatenate([actual, forecast_values]), 0.995))
    limit = max(limit, 1e-12)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, models, strict=True):
        group = forecasts.loc[forecasts["model"].eq(model)]
        axis.scatter(group["forecast"], group["actual"], s=14, alpha=0.45)
        axis.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_title(model.replace("_", " "))
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast five-session realized variance")
    for axis in axes[:, 0]:
        axis.set_ylabel("Actual five-session realized variance")
    figure.suptitle("Actual versus forecast on common Stage 3 origins")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _package_versions() -> str:
    """List versions used to generate the analysis."""
    packages = ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")
    values = [f"Python {platform.python_version()}"]
    for package in packages:
        try:
            values.append(f"{package} {version(package)}")
        except PackageNotFoundError:
            values.append(f"{package} version unavailable")
    return ", ".join(values)


def _write_summary(
    summary: pd.DataFrame,
    dm_results: pd.DataFrame,
    calibration: pd.DataFrame,
    iv_results: pd.DataFrame,
    iv_metadata: dict[str, object],
    residuals: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> None:
    """Write the result and explain what can and cannot be concluded."""
    sorted_rmse = summary.sort_values("RMSE")
    best_rmse = sorted_rmse.iloc[0]
    best_mae = summary.sort_values("MAE").iloc[0]
    best_qlike = summary.sort_values("QLIKE").iloc[0]
    primary = dm_results.loc[
        dm_results["benchmark"].eq(BENCHMARK)
        & dm_results["loss"].eq("squared_error")
        & dm_results["bandwidth"].eq(HORIZON - 1)
    ]
    primary_qlike = dm_results.loc[
        dm_results["benchmark"].eq(BENCHMARK)
        & dm_results["loss"].eq("QLIKE")
        & dm_results["bandwidth"].eq(HORIZON - 1)
    ]
    significant_better = primary.loc[
        (primary["mean_loss_difference_candidate_minus_benchmark"] < 0)
        & (primary["p_holm"] < 0.05),
        "candidate",
    ].tolist()
    significant_worse = primary.loc[
        (primary["mean_loss_difference_candidate_minus_benchmark"] > 0)
        & (primary["p_holm"] < 0.05),
        "candidate",
    ].tolist()
    qlike_significant_better = primary_qlike.loc[
        (primary_qlike["mean_loss_difference_candidate_minus_benchmark"] < 0)
        & (primary_qlike["p_holm"] < 0.05),
        "candidate",
    ].tolist()
    qlike_significant_worse = primary_qlike.loc[
        (primary_qlike["mean_loss_difference_candidate_minus_benchmark"] > 0)
        & (primary_qlike["p_holm"] < 0.05),
        "candidate",
    ].tolist()
    biased = calibration.loc[calibration["mean_error_hac_p"] < 0.05]
    iv_raw = iv_results.loc[iv_results["model"].eq("xau_1m_iv_unscaled")].iloc[0]
    iv_scaled = iv_results.loc[iv_results["model"].eq("xau_1m_iv_scaled")].iloc[0]
    project_models = calibration.loc[
        ~calibration["model"].str.startswith("xau_1m_iv_")
    ]
    mz_rejected = project_models.loc[
        project_models["mz_joint_test_p"] < 0.05, "model"
    ].tolist()
    vix_comparison = dm_results.loc[
        dm_results["candidate"].eq("garchx_vix")
        & dm_results["benchmark"].eq("garch11")
        & dm_results["loss"].eq("squared_error")
    ]
    if significant_better:
        better_text = ", ".join(significant_better)
    else:
        better_text = "None after Holm correction"
    if significant_worse:
        worse_text = ", ".join(significant_worse)
    else:
        worse_text = "None after Holm correction"
    qlike_better_text = (
        ", ".join(qlike_significant_better)
        if qlike_significant_better
        else "None after Holm correction"
    )
    qlike_worse_text = (
        ", ".join(qlike_significant_worse)
        if qlike_significant_worse
        else "None after Holm correction"
    )
    if biased.empty:
        bias_text = "No model has a HAC mean-error test with p < 0.05."
    else:
        bias_text = ", ".join(
            f"{row.model} (mean error {row.mean_error_actual_minus_forecast:.3g}, "
            f"p={row.mean_error_hac_p:.3g})"
            for row in biased.itertuples()
        )
    if mz_rejected:
        mz_text = (
            f"The HAC Mincer-Zarnowitz joint test rejects (intercept=0, slope=1) "
            f"for {len(mz_rejected)} of {len(project_models)} models at 5%: "
            + ", ".join(mz_rejected)
            + "."
        )
    else:
        mz_text = (
            f"The HAC Mincer-Zarnowitz joint test does not reject the ideal "
            f"intercept and slope for any of the {len(project_models)} models."
        )
    if not vix_comparison.empty:
        vix_row = vix_comparison.iloc[0]
        vix_text = (
            f"Against GARCH(1,1), `garchx_vix` has mean squared-loss difference "
            f"{vix_row['mean_loss_difference_candidate_minus_benchmark']:.3g} "
            f"(HLN p={vix_row['p_raw']:.3g}). VIX is an input to this model, "
            "not a standalone gold-volatility forecast."
        )
    else:
        vix_text = "No direct VIX-model comparison was available."
    significant_residual = residuals.loc[
        (residuals["series"].eq("error")) & (residuals["p_value"] < 0.05)
    ]
    residual_text = (
        "Some leading-model errors show Ljung-Box evidence of serial dependence "
        "at one or more tested lags; overlapping targets make this diagnostic exploratory."
        if not significant_residual.empty
        else "The tested leading-model errors show no Ljung-Box p-value below 0.05; "
        "this does not prove independence."
    )
    first_origin = forecasts["origin_date"].min().date().isoformat()
    last_origin = forecasts["origin_date"].max().date().isoformat()
    model_count = forecasts["model"].nunique()
    lines = [
        "# Samuel — model comparison",
        "",
        "## Scope",
        "",
        "**This analysis uses the Stage 3 forecasts currently in the workspace, not the Yang–Zhang Stage 4 files.**",
        "The available target is five-session realized variance from the Yahoo `GC=F` continuous gold-futures series.",
        "The requested Stage 4 Yang–Zhang forecast files are not present in this checkout, so no result below should be presented as a Yang–Zhang result.",
        "",
        f"- Models: {model_count}; common forecast origins: {len(forecasts) // model_count}.",
        f"- Evaluation period: {first_origin} to {last_origin}.",
        f"- Horizon: {HORIZON} observed sessions; target windows overlap.",
        f"- Primary ranking: RMSE (equivalent ordering to MSE); QLIKE and MAE are additional criteria.",
        f"- HAC lag/bandwidth for this five-session target: {HAC_LAGS}.",
        "",
        "## Results",
        "",
        f"- Lowest RMSE: **{best_rmse['model']}** (RMSE {best_rmse['RMSE']:.6g}; MAE {best_rmse['MAE']:.6g}; QLIKE {best_rmse['QLIKE']:.4f}).",
        f"- Lowest MAE: **{best_mae['model']}** (MAE {best_mae['MAE']:.6g}).",
        f"- Lowest QLIKE: **{best_qlike['model']}** (QLIKE {best_qlike['QLIKE']:.4f}).",
        f"- Compared with expanding historical variance under squared-error loss, models with significantly lower loss after Holm correction: {better_text}.",
        f"- Models with significantly higher squared-error loss after Holm correction: {worse_text}.",
        f"- Under QLIKE, significantly lower loss after Holm correction: {qlike_better_text}; significantly higher loss: {qlike_worse_text}.",
        f"- Mean-error HAC bias tests (actual minus forecast; p-values are unadjusted): {bias_text}",
        f"- Mincer-Zarnowitz calibration: {mz_text} Failure to reject is not proof of perfect calibration.",
        f"- Residual checks: {residual_text}",
        f"- VIX role: {vix_text}",
        "",
        "## Implied-volatility reference",
        "",
        f"- Gold 1-month implied volatility was converted using `{iv_metadata['iv_conversion']}`.",
        "- Quotes were aligned using the previous available quote; same-day quotes were excluded.",
        f"- The scale factor ({iv_metadata['iv_scale_factor']:.6g}) uses {iv_metadata['scale_training_n']} pre-test targets whose target windows had ended before the first test origin ({iv_metadata['first_test_origin']}).",
        f"- On {int(iv_raw['n'])} dates with a prior quote, unscaled IV RMSE/MAE were {iv_raw['RMSE']:.6g}/{iv_raw['MAE']:.6g}; training-scaled IV RMSE/MAE were {iv_scaled['RMSE']:.6g}/{iv_scaled['MAE']:.6g}.",
        f"- Nominal DM comparisons with the test-sample RMSE winner give Holm-adjusted squared-error p-values {iv_raw['dm_vs_test_rmse_winner_mse_p_holm']:.3g} (unscaled) and {iv_scaled['dm_vs_test_rmse_winner_mse_p_holm']:.3g} (scaled); the winner was selected on this same test sample, so these are exploratory.",
        f"- The raw IV RMSE is lower than the top model's RMSE ({iv_raw['RMSE']:.6g} versus {best_rmse['RMSE']:.6g}), but the Holm-adjusted DM p-value is {iv_raw['dm_vs_test_rmse_winner_mse_p_holm']:.3g}; this does not meet the 5% threshold.",
        (
            f"- Quote quality flag: {iv_metadata['quotes_below_1pct_in_test']} test quote(s) had MID_PRICE below 1% "
            f"({iv_metadata['low_quote_rows']}). The raw unscaled IV QLIKE is {iv_raw['QLIKE']:.4f}; "
            f"excluding that flagged date only as a sensitivity check changes it to "
            f"{iv_raw['QLIKE_sensitivity_excluding_quotes_below_1pct']:.4f}. "
            "The original quote is retained; its source should be verified before treating IV QLIKE as reliable."
            if iv_metadata["quotes_below_1pct_in_test"]
            else "- No test quote was below the 1% screening threshold."
        ),
        "- This is only an exploratory reference: the implied-volatility quote is one-month, while the evaluated target is five-session futures realized variance; constant variance over the five sessions is assumed for conversion, and the implied-volatility underlying may not match the futures series.",
        "- VIX measures U.S. equity implied volatility, not gold implied volatility. It is treated as a model predictor here, not as a direct gold forecast benchmark.",
        "",
        "## Limitations",
        "",
        "1. The available data do not match the proposed next-valid-day Yang–Zhang target; the requested YZ forecast artifacts are absent.",
        "2. The target is a noisy five-session realized-variance proxy, and adjacent target windows overlap, creating dependent losses.",
        "3. `GC=F` is a continuous futures series, not spot gold or a single fixed-expiry contract; results depend on that instrument and its roll behavior.",
        "4. The implied-volatility benchmark has a one-month maturity and potentially different underlying/instrument; converting it to five-session variance requires a flat-variance assumption.",
        "5. The same test sample is used to rank models and identify the RMSE winner; post-selection comparisons are exploratory, and the test period does not establish future stability or causality.",
        "",
        "## Reproducibility",
        "",
        f"- Random seed set to {SEED}.",
        f"- Package versions: {_package_versions()}.",
        "- Source forecasts and input data were read only; this analysis writes new files under `outputs/samuel/`.",
        "- DM comparisons use Newey–West Bartlett long-run variance, Harvey–Leybourne–Newbold correction, and Holm correction separately by loss and bandwidth.",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the saved-forecast comparison and write new analysis outputs."""
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Validate the saved forecasts and summarize their out-of-sample losses.
    forecasts, panel, implied_vol = _read_inputs()
    summary = _losses(forecasts)
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Step 2: Test candidate losses against historical variance with HAC and Holm adjustment.
    dm_results = _dm_tests(forecasts)
    dm_results.to_csv(OUTPUT_DIR / "dm_results.csv", index=False)

    # Step 3: Estimate calibration and mean-error bias using HAC standard errors.
    calibration = _calibration(forecasts)

    # Step 4: Align gold implied volatility to prior quotes and the common test origins.
    iv_results, iv_metadata, implied_test = _implied_volatility(
        forecasts, panel, implied_vol, summary
    )
    iv_results.to_csv(OUTPUT_DIR / "iv_benchmark.csv", index=False)
    iv_calibration_input = []
    for name, column in (
        ("xau_1m_iv_unscaled", "iv_5session_variance"),
        ("xau_1m_iv_scaled", "iv_5session_variance_scaled"),
    ):
        iv_calibration_input.append(
            implied_test[["origin_date", "actual", column]]
            .rename(columns={column: "forecast"})
            .assign(model=name)
        )
    calibration_input = pd.concat(
        [forecasts, *iv_calibration_input],
        ignore_index=True,
        sort=False,
    )
    calibration = pd.concat(
        [calibration, _calibration(calibration_input.loc[calibration_input["model"].isin(
            ["xau_1m_iv_unscaled", "xau_1m_iv_scaled"]
        )])],
        ignore_index=True,
    )
    calibration.to_csv(OUTPUT_DIR / "calibration.csv", index=False)

    # Step 5: Check residual dependence and plot leading models against actual values.
    residuals = _residual_diagnostics(forecasts, implied_test, summary)
    residuals.to_csv(OUTPUT_DIR / "residual_diagnostics.csv", index=False)
    _plot_top_models(
        forecasts,
        summary,
        OUTPUT_DIR / "top_models_actual_vs_forecast.png",
    )

    # Step 6: Write a plain-English summary with the scope and main limitations.
    _write_summary(
        summary,
        dm_results,
        calibration,
        iv_results,
        iv_metadata,
        residuals,
        forecasts,
    )
    print(summary.head(10).to_string(index=False))
    print("\nDM tests against historical variance, five-session bandwidth:")
    print(
        dm_results.loc[
            dm_results["benchmark"].eq(BENCHMARK)
            & dm_results["bandwidth"].eq(HORIZON - 1)
            & dm_results["loss"].eq("squared_error")
        ][["candidate", "mean_loss_difference_candidate_minus_benchmark", "p_raw", "p_holm"]]
        .sort_values("p_holm")
        .to_string(index=False)
    )
    print("\nImplied-volatility benchmark:")
    print(iv_results[["model", "n", "MAE", "RMSE", "QLIKE"]].to_string(index=False))
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
