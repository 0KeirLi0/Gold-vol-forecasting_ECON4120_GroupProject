"""Select one HK order and refit it on aligned Stage 3 forecast origins."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gold_vol_pipeline.auto_arima import fit_arima_order, hyndman_khandakar
from gold_vol_pipeline.evaluation import summary_table

HORIZON = 5
MODEL = "auto_arima_hk"
PANEL = REPO_ROOT / "data/processed/stage2_multivariate.csv"
COMBINED = REPO_ROOT / "data/processed/evaluation_forecasts_with_stage1.csv"
BASELINES = REPO_ROOT / "data/processed/stage1_on_stage3_origins_forecasts.csv"
OUTPUT = REPO_ROOT / "data/processed/stage3_auto_arima_forecasts.csv"


def forecast_origin(task: tuple[pd.Timestamp, np.ndarray, float, tuple[int, int, int]]) -> dict:
    origin, returns, actual, order = task
    try:
        fit = fit_arima_order(returns**2, order)
        daily_forecasts = np.asarray(fit.get_forecast(steps=HORIZON).predicted_mean)
        forecast = float(np.maximum(daily_forecasts, 0).sum())
        if not np.isfinite(forecast) or forecast < 0:
            raise RuntimeError("HK auto-ARIMA returned an invalid forecast.")
        return {
            "origin_date": origin,
            "model": MODEL,
            "forecast": forecast,
            "actual": actual,
            "converged": True,
            "error": "",
            "order": str(order),
            "n_returns": len(returns),
        }
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
        return {
            "origin_date": origin,
            "model": MODEL,
            "forecast": np.nan,
            "actual": actual,
            "converged": False,
            "error": str(error),
            "order": str(order),
            "n_returns": len(returns),
        }


def prepare_tasks(panel: pd.DataFrame, forecasts: pd.DataFrame) -> list:
    required_panel = {"date", "log_return", "realized_variance"}
    required_forecasts = {"origin_date", "model", "forecast", "actual"}
    if not required_panel.issubset(panel.columns):
        raise ValueError(f"Panel must contain {sorted(required_panel)}.")
    if not required_forecasts.issubset(forecasts.columns):
        raise ValueError(f"Forecasts must contain {sorted(required_forecasts)}.")
    if panel["date"].isna().any() or panel["date"].duplicated().any():
        raise ValueError("Panel dates must be nonmissing and unique.")
    if forecasts["origin_date"].isna().any() or forecasts.duplicated(
        ["origin_date", "model"]
    ).any():
        raise ValueError("Forecast origins must be nonmissing and unique per model.")
    panel = panel.sort_values("date").set_index("date")
    origins = pd.DatetimeIndex(sorted(forecasts["origin_date"].unique()))
    if not origins.isin(panel.index).all():
        raise ValueError("Forecast origins are missing from the panel.")
    actuals = pd.to_numeric(forecasts["actual"], errors="raise")
    expected = panel["realized_variance"].reindex(forecasts["origin_date"])
    if not np.isfinite(actuals).all() or not np.allclose(
        actuals.to_numpy(), expected.to_numpy(dtype=float), rtol=0, atol=1e-14
    ):
        raise ValueError("Saved forecast actuals do not match the panel.")
    tasks = []
    for origin in origins:
        returns = panel.loc[:origin, "log_return"].dropna().to_numpy(dtype=float)
        tasks.append((origin, returns, float(panel.at[origin, "realized_variance"])))
    return tasks


def make_comparison(forecasts: pd.DataFrame, results: pd.DataFrame):
    combined = pd.concat(
        [forecasts.loc[forecasts["model"].ne(MODEL)], results],
        ignore_index=True,
        sort=False,
    ).sort_values(["origin_date", "model"])
    origins_by_model = [
        set(group.loc[group["forecast"].notna(), "origin_date"])
        for _, group in combined.groupby("model")
    ]
    common_origins = set.intersection(*origins_by_model)
    if not common_origins:
        raise ValueError("No complete common-origin sample remains.")
    common = combined.loc[
        combined["origin_date"].isin(common_origins) & combined["forecast"].notna()
    ].copy()
    counts = common.groupby("model").size()
    if len(counts) != combined["model"].nunique() or not counts.eq(len(common_origins)).all():
        raise ValueError("The common-origin comparison is not balanced.")
    summaries = []
    for frame in (combined, common):
        summary = summary_table(
            frame, benchmark="historical_variance", horizon=HORIZON, top_n=5
        )
        summary["RMSE"] = np.sqrt(summary["MSE"])
        summaries.append(summary)
    return combined, common, summaries[0], summaries[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    forecasts = pd.read_csv(COMBINED, parse_dates=["origin_date"])
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    tasks = prepare_tasks(panel, forecasts)
    if len(tasks) != 445:
        raise ValueError(f"Expected 445 Stage 3 origins, found {len(tasks)}.")
    selected = hyndman_khandakar(tasks[0][1] ** 2)
    order = selected.order
    print(f"Selected HK order {order} at {tasks[0][0].date()}; "
          f"refitting at {len(tasks)} origins.", flush=True)
    fixed_tasks = [(origin, returns, actual, order) for origin, returns, actual in tasks]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = []
        for row in executor.map(forecast_origin, fixed_tasks):
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f"Completed {len(rows)}/{len(tasks)} origins.", flush=True)
    results = pd.DataFrame(rows)
    combined, common, summary, common_summary = make_comparison(forecasts, results)
    baseline_forecasts = pd.read_csv(BASELINES, parse_dates=["origin_date"])
    baseline_forecasts = pd.concat(
        [baseline_forecasts.loc[baseline_forecasts["model"].ne(MODEL)], results],
        ignore_index=True,
        sort=False,
    ).sort_values(["origin_date", "model"])
    baseline_summary = common_summary.loc[
        common_summary["model"].isin(baseline_forecasts["model"].unique())
    ]
    metric_columns = [
        "model", "n", "MSE", "RMSE", "MAE", "QLIKE", "DM_MSE_p_holm",
        "DM_QLIKE_p_holm", "MZ_a", "MZ_b", "MZ_wald_p",
    ]
    results.to_csv(OUTPUT, index=False)
    baseline_forecasts.to_csv(BASELINES, index=False)
    baseline_summary[metric_columns].to_csv(
        BASELINES.with_name("stage1_on_stage3_origins_metrics.csv"), index=False
    )
    combined.to_csv(COMBINED, index=False)
    common.to_csv(COMBINED.with_name(
        "evaluation_forecasts_with_stage1_common_origins.csv"), index=False)
    summary.to_csv(COMBINED.with_name("evaluation_summary_with_stage1.csv"), index=False)
    common_summary.to_csv(COMBINED.with_name(
        "evaluation_summary_with_stage1_common_origins.csv"), index=False)
    manifest = {
        "evaluation": "HK order selected at first origin and refit at every Stage 3 origin",
        "panel_sha256": hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        "horizon": HORIZON,
        "search": {"max_p": 5, "max_q": 5, "max_d": 2, "max_order": 5, "alpha": 0.05},
        "selection_origin": tasks[0][0].date().isoformat(),
        "selection_n_returns": len(tasks[0][1]),
        "selected_order": list(order),
        "selection_aicc": selected.aicc,
        "selection_kpss_p_values": list(selected.differencing_p_values),
        "origin_count": len(tasks),
        "successful_forecasts": int(results["forecast"].notna().sum()),
        "failed_forecasts": int(results["forecast"].isna().sum()),
        "common_complete_model_origins": int(common["origin_date"].nunique()),
        "orders": results["order"].value_counts().to_dict(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.with_name("stage3_auto_arima_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    baseline_manifest_path = BASELINES.with_name("stage1_on_stage3_origins_manifest.json")
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    baseline_manifest["models"] = list(baseline_forecasts["model"].drop_duplicates())
    baseline_manifest["common_complete_model_origins"] = manifest["common_complete_model_origins"]
    baseline_manifest["common_origin_range"] = {
        "start": common["origin_date"].min().date().isoformat(),
        "end": common["origin_date"].max().date().isoformat(),
    }
    baseline_manifest["auto_arima_evaluation"] = "stage3_auto_arima_manifest.json"
    baseline_manifest_path.write_text(
        json.dumps(baseline_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    print(common_summary.loc[common_summary["model"].eq(MODEL)].to_string(index=False))


if __name__ == "__main__":
    main()
