"""Holdout-split, spike, window, and seasonality robustness for the Group 2 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gold_vol_pipeline.report_diagnostics import (
    acf_pacf_table,
    assign_forecast_quintiles,
    dickey_fuller_stationary_alternative,
    evaluate_harx_month_dummies,
    q5_influence_table,
    shortlist_display_name,
    subperiod_metrics,
    train_fraction_for_cutoff,
)
from gold_vol_pipeline.volatility_forecasting import evaluate_daily_volatility
from gold_vol_pipeline.yang_zhang import prepare_yang_zhang_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ERRORS = ROOT / "report_group2" / "diagnostics" / "holdout_forecast_errors.csv"
DEFAULT_TARGET = ROOT / "data" / "processed" / "GCc1_yang_zhang_volatility.csv"
DEFAULT_OHLC = ROOT / "data" / "refinitiv" / "GCc1.csv"
DEFAULT_OUTPUT = ROOT / "report_group2" / "diagnostics"
DEFAULT_FIGURES = ROOT / "report_group2" / "figures"
LAST_TRAIN_DATE = "2023-06-01"
WINDOW_MODELS = ("yang_zhang_persistence", "garch11", "harx_ohlc")
SUBPERIODS = {
    "full_holdout": ("2023-06-02", "2026-11-10"),
    "development_2023_2024": ("2023-06-02", "2024-12-31"),
    "confirmatory_2025_2026": ("2025-01-01", "2026-11-10"),
    "pre_spike": ("2023-06-02", "2026-01-29"),
    "spike_cluster": ("2026-01-30", "2026-03-31"),
}


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_spike_event_table(ohlc_path: Path, target_path: Path, output: Path) -> pd.DataFrame:
    raw = pd.read_csv(ohlc_path)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    target = pd.read_csv(target_path, parse_dates=["date"])
    yz = target.loc[:, ["date", "daily_volatility", "complete_ohlc"]].rename(
        columns={"date": "Date"}
    )
    merged = raw.merge(yz, on="Date", how="left")
    windows = [
        ("crash_window", "2026-01-26", "2026-02-05"),
        ("march_drawdown", "2026-03-20", "2026-03-25"),
    ]
    rows = []
    for label, start, end in windows:
        block = merged.loc[merged["Date"].between(start, end)].copy()
        block["event_window"] = label
        rows.append(block)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(output, index=False)
    return table


def write_market_event_notes(output: Path) -> pd.DataFrame:
    notes = pd.DataFrame(
        [
            {
                "date": "2026-01-29",
                "source": "Reuters; Irish Times; CNBC contemporaneous coverage",
                "note": (
                    "Gold futures printed a record near 5,595 USD/oz and then reversed. "
                    "GCc1 that session: high 5586.2, low 5097.5, close 5364.7. "
                    "Open interest jumps from 171 on 26 Jan to 30,082 on 29 Jan "
                    "(generic continuation rolling onto the liquid contract)."
                ),
            },
            {
                "date": "2026-01-30",
                "source": "Reuters 2026-02-02; CNBC",
                "note": (
                    "Steepest daily gold drop since 1983 after the nomination of "
                    "Kevin Warsh as Federal Reserve chair. GCc1 high 5440.5, low 4700.0, "
                    "close 4879.6. Yang-Zhang is missing: the n=2 window still contains "
                    "the incomplete 27-28 Jan bars through the overnight return."
                ),
            },
            {
                "date": "2026-02-02",
                "source": "Reuters 2026-02-02, 'Gold, silver extend fall as CME margin hikes compound sharp selloff'",
                "note": (
                    "Selloff continues; CME raised gold maintenance margins from 6% to 8%. "
                    "GCc1 low prints exactly 4400.0. First valid n=2 Yang-Zhang after the "
                    "crash equals 0.077521, the sample maximum. This is a delayed OHLC "
                    "estimator after missing bars, not a single-tick glitch."
                ),
            },
            {
                "date": "2026-03-23",
                "source": "Contemporaneous market reports of the Q1 2026 gold drawdown",
                "note": (
                    "Second-largest Yang-Zhang in the prepared sample (0.061318), "
                    "with GCc1 low 4100.8, during the later peak-to-trough drawdown."
                ),
            },
        ]
    )
    notes.to_csv(output, index=False)
    return notes


def economic_mae_illustration(errors: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
    hgb = errors.loc[errors["model"].eq("hist_gradient_boosting_ohlc"), "forecast_error"]
    hgb_mae = float(np.mean(np.abs(hgb)))
    persistence_mae = 0.003190
    gap = persistence_mae - hgb_mae
    holdout = target.loc[target["split"].eq("test")].copy()
    if holdout.empty:
        holdout = target.loc[
            pd.to_datetime(target["date"]).between("2023-06-02", "2026-11-10")
        ]
    price = pd.to_numeric(holdout["close"], errors="coerce")
    mean_price = float(price.mean())
    last_price = float(price.dropna().iloc[-1])
    annualised_gap = gap * float(np.sqrt(252))
    return {
        "hgb_mae": hgb_mae,
        "persistence_mae": persistence_mae,
        "mae_gap": gap,
        "mae_gap_pct_of_persistence": gap / persistence_mae,
        "annualised_vol_point_gap": annualised_gap,
        "holdout_mean_price": mean_price,
        "holdout_last_price": last_price,
        "var_1d_1p65_usd_per_oz_at_mean_price": 1.65 * gap * mean_price,
        "comex_100oz_usd_at_mean_price": 1.65 * gap * mean_price * 100,
        "var_1d_1p65_usd_per_oz_at_last_price": 1.65 * gap * last_price,
        "comex_100oz_usd_at_last_price": 1.65 * gap * last_price * 100,
    }


def plot_q5_influence(table: pd.DataFrame, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10.6, 4.6), constrained_layout=True)
    models = list(dict.fromkeys(table["model"]))
    x = np.arange(len(models))
    width = 0.24
    for offset, k in enumerate((1, 3, 5)):
        block = table.loc[table["k_dropped"].eq(k)].set_index("model").loc[models]
        axis.bar(
            x + (offset - 1) * width,
            100 * block["q5_variance_decline"].to_numpy(dtype=float),
            width=width,
            label=f"Drop top {k}",
        )
    axis.set_xticks(x)
    axis.set_xticklabels([shortlist_display_name(model) for model in models], rotation=20)
    axis.set_ylabel("Q5 residual-variance decline (%)")
    axis.set_title("How much forecast-Q5 variance is carried by the largest misses")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, path)


def plot_subperiod_mae(table: pd.DataFrame, path: Path) -> Path:
    keep = table.loc[
        table["split"].isin(
            ["development_2023_2024", "confirmatory_2025_2026", "full_holdout"]
        )
    ].copy()
    figure, axis = plt.subplots(figsize=(10.6, 4.6), constrained_layout=True)
    models = list(dict.fromkeys(keep["model"]))
    x = np.arange(len(models))
    width = 0.24
    labels = {
        "development_2023_2024": "2023-06 to 2024-12",
        "confirmatory_2025_2026": "2025-01 to 2026-11",
        "full_holdout": "Full 761-day holdout",
    }
    for offset, split in enumerate(
        ("development_2023_2024", "confirmatory_2025_2026", "full_holdout")
    ):
        block = keep.loc[keep["split"].eq(split)].set_index("model").loc[models]
        axis.bar(
            x + (offset - 1) * width,
            block["MAE"].to_numpy(dtype=float),
            width=width,
            label=labels[split],
        )
    axis.set_xticks(x)
    axis.set_xticklabels([shortlist_display_name(model) for model in models], rotation=20)
    axis.set_ylabel("MAE")
    axis.set_title("Holdout MAE on the 2023-24 development window versus 2025-26")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, path)


def plot_window_acf(acf_table: pd.DataFrame, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10.2, 4.4), constrained_layout=True)
    for window, block in acf_table.groupby("window"):
        train = block.loc[block["sample"].eq("train") & block["lag"].gt(0)]
        axis.plot(
            train["lag"],
            train["acf"],
            marker="o",
            markersize=3,
            linewidth=1.1,
            label=f"n={window}",
        )
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set(
        xlabel="Lag (sessions)",
        ylabel="ACF of Yang-Zhang volatility",
        title="Training ACF by Yang-Zhang window",
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, path)


def window_target_diagnostics(
    ohlc_path: Path,
    windows: tuple[int, ...],
    last_train_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    adf_rows = []
    acf_rows = []
    prepared_by_window: dict[int, pd.DataFrame] = {}
    cutoff = pd.Timestamp(last_train_date)
    for window in windows:
        prepared = prepare_yang_zhang_dataset(ohlc_path, window=window)
        valid = prepared.loc[prepared["daily_volatility"].notna()].copy()
        valid["date"] = pd.to_datetime(valid["date"])
        train = valid.loc[valid["date"] <= cutoff, "daily_volatility"]
        adf = dickey_fuller_stationary_alternative(
            train, name="yang_zhang", sample="train"
        )
        adf["window"] = window
        adf["n_train"] = int(len(train))
        adf["n_test"] = int((valid["date"] > cutoff).sum())
        adf["n_valid"] = int(len(valid))
        adf["train_fraction"] = train_fraction_for_cutoff(valid["date"], cutoff)
        adf_rows.append(adf)
        acf_table = acf_pacf_table(
            train, name="yang_zhang", sample="train", nlags=30
        )
        acf_table["window"] = window
        acf_rows.append(acf_table)
        prepared_by_window[window] = prepared
    return pd.DataFrame(adf_rows), pd.concat(acf_rows, ignore_index=True), prepared_by_window


def summarize_window_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, model), group in forecasts.groupby(["window", "model"], sort=False):
        successful = group.loc[group["converged"] & np.isfinite(group["forecast"])].copy()
        err = successful["actual"] - successful["forecast"]
        successful["forecast_quintile"] = assign_forecast_quintiles(successful["forecast"])
        q5 = successful.loc[successful["forecast_quintile"].eq(5), "actual"] - successful.loc[
            successful["forecast_quintile"].eq(5), "forecast"
        ]
        q1 = successful.loc[successful["forecast_quintile"].eq(1), "actual"] - successful.loc[
            successful["forecast_quintile"].eq(1), "forecast"
        ]
        rows.append(
            {
                "window": int(window),
                "model": str(model),
                "n": int(len(successful)),
                "failure_count": int((~group["converged"].astype(bool)).sum()),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(np.square(err)))),
                "mean_error": float(np.mean(err)),
                "q5_variance": float(np.var(q5, ddof=1)) if len(q5) > 1 else np.nan,
                "q1_variance": float(np.var(q1, ddof=1)) if len(q1) > 1 else np.nan,
                "q5_q1_ratio": (
                    float(np.var(q5, ddof=1) / np.var(q1, ddof=1))
                    if len(q5) > 1 and len(q1) > 1
                    else np.nan
                ),
                "acf_lag1_errors": float(err.autocorr(lag=1)),
            }
        )
    return pd.DataFrame(rows)


def run_window_forecasts(
    prepared_by_window: dict[int, pd.DataFrame],
    last_train_date: str,
    workers: int,
) -> pd.DataFrame:
    frames = []
    cutoff = pd.Timestamp(last_train_date)
    for window, prepared in prepared_by_window.items():
        valid = prepared.loc[prepared["daily_volatility"].notna()].copy()
        valid["date"] = pd.to_datetime(valid["date"])
        fraction = train_fraction_for_cutoff(valid["date"], cutoff)
        print(
            f"Window n={window}: {int((valid['date'] <= cutoff).sum())} train / "
            f"{int((valid['date'] > cutoff).sum())} test, train_fraction={fraction:.6f}",
            flush=True,
        )
        results, _summary, _manifest = evaluate_daily_volatility(
            prepared,
            train_fraction=fraction,
            model_names=WINDOW_MODELS,
            workers=workers,
            progress=True,
        )
        results = results.copy()
        results["window"] = window
        frames.append(results)
    return pd.concat(frames, ignore_index=True)


def ranking_stability(subperiods: pd.DataFrame) -> pd.DataFrame:
    development = subperiods.loc[subperiods["split"].eq("development_2023_2024")].copy()
    confirmatory = subperiods.loc[subperiods["split"].eq("confirmatory_2025_2026")].copy()
    development["rank_mae"] = development["MAE"].rank(method="min")
    confirmatory["rank_mae"] = confirmatory["MAE"].rank(method="min")
    merged = development.merge(
        confirmatory,
        on="model",
        suffixes=("_dev", "_conf"),
    )
    return merged[
        [
            "model",
            "n_dev",
            "MAE_dev",
            "rank_mae_dev",
            "q5_q1_ratio_dev",
            "n_conf",
            "MAE_conf",
            "rank_mae_conf",
            "q5_q1_ratio_conf",
        ]
    ]


def summarize_month_dummy(forecasts: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    harx = baseline.loc[baseline["model"].eq("harx_ohlc")].copy()
    common = harx.merge(
        forecasts.loc[forecasts["converged"]],
        on="target_date",
        suffixes=("_harx", "_month"),
    )
    rows = []
    for label, forecast_col, actual_col in (
        ("harx_ohlc", "forecast_harx", "actual_harx"),
        ("harx_month_dummies", "forecast_month", "actual_month"),
    ):
        err = common[actual_col] - common[forecast_col]
        quintile_source = common[forecast_col]
        labels = assign_forecast_quintiles(quintile_source)
        q5 = err.loc[labels.eq(5)]
        q1 = err.loc[labels.eq(1)]
        rows.append(
            {
                "model": label,
                "n_common": int(len(common)),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(np.square(err)))),
                "mean_error": float(np.mean(err)),
                "q5_variance": float(np.var(q5, ddof=1)),
                "q1_variance": float(np.var(q1, ddof=1)),
                "q5_q1_ratio": float(np.var(q5, ddof=1) / np.var(q1, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robustness tables for holdout reuse, spikes, windows, and seasonality."
    )
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--ohlc", type=Path, default=DEFAULT_OHLC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Yang-Zhang windows for ACF/ADF. Sequential forecasts skip n=2 by default.",
    )
    parser.add_argument(
        "--skip-sequential",
        action="store_true",
        help="Write influence, subperiod, spike, and ACF/ADF tables only.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    errors = pd.read_csv(args.errors, parse_dates=["target_date", "origin_date"])
    target = pd.read_csv(args.target, parse_dates=["date"])
    influence = q5_influence_table(errors)
    influence["display_name"] = influence["model"].map(shortlist_display_name)
    influence.to_csv(args.output_dir / "q5_error_influence.csv", index=False)
    subperiods = subperiod_metrics(errors, splits=SUBPERIODS)
    subperiods["display_name"] = subperiods["model"].map(shortlist_display_name)
    subperiods.to_csv(args.output_dir / "holdout_subperiod_metrics.csv", index=False)
    ranks = ranking_stability(subperiods)
    ranks["display_name"] = ranks["model"].map(shortlist_display_name)
    ranks.to_csv(args.output_dir / "holdout_ranking_stability.csv", index=False)
    write_spike_event_table(
        args.ohlc, args.target, args.output_dir / "spike_ohlc_window.csv"
    )
    write_market_event_notes(args.output_dir / "spike_market_events.csv")
    economics = economic_mae_illustration(errors, target)
    (args.output_dir / "economic_mae_illustration.json").write_text(
        json.dumps(economics, indent=2) + "\n", encoding="utf-8"
    )
    plot_q5_influence(influence, args.figure_dir / "q5_error_influence.png")
    plot_subperiod_mae(subperiods, args.figure_dir / "holdout_subperiod_mae.png")

    windows = tuple(args.windows)
    adf_table, acf_table, prepared_by_window = window_target_diagnostics(
        args.ohlc, windows, LAST_TRAIN_DATE
    )
    adf_table.to_csv(args.output_dir / "window_adf.csv", index=False)
    acf_table.to_csv(args.output_dir / "window_acf.csv", index=False)
    plot_window_acf(acf_table, args.figure_dir / "window_acf.png")

    if args.skip_sequential:
        print("Skipped sequential window and month-dummy forecasts.", flush=True)
        print(influence.to_string(index=False), flush=True)
        print(ranks.to_string(index=False), flush=True)
        return 0

    sequential_windows = {
        window: prepared
        for window, prepared in prepared_by_window.items()
        if window != 2
    }
    if sequential_windows:
        window_forecasts = run_window_forecasts(
            sequential_windows, LAST_TRAIN_DATE, args.workers
        )
        window_forecasts.to_csv(
            args.output_dir / "window_sensitivity_forecasts.csv", index=False
        )
        window_summary = summarize_window_forecasts(window_forecasts)
        # Add n=2 reuse of the saved shortlist for persistence-comparable models.
        n2 = errors.loc[errors["model"].isin(["garch11", "harx_ohlc"])].copy()
        n2["window"] = 2
        n2["converged"] = True
        n2_summary = summarize_window_forecasts(n2)
        window_summary = pd.concat([n2_summary, window_summary], ignore_index=True)
        window_summary.to_csv(
            args.output_dir / "window_sensitivity_summary.csv", index=False
        )

    print("Fitting HAR-X with next-session month dummies...", flush=True)
    prepared = prepare_yang_zhang_dataset(args.ohlc, window=2)
    month_forecasts = evaluate_harx_month_dummies(
        prepared, last_train_date=LAST_TRAIN_DATE
    )
    month_forecasts.to_csv(args.output_dir / "harx_month_dummy_forecasts.csv", index=False)
    month_summary = summarize_month_dummy(month_forecasts, errors)
    month_summary.to_csv(args.output_dir / "harx_month_dummy_summary.csv", index=False)
    print(month_summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
