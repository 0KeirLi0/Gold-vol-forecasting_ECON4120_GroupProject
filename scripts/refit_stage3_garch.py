"""Refit scaled GARCH on all Stage 3 origins and refresh aligned comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gold_vol_pipeline.evaluation import summary_table
from gold_vol_pipeline.models import garch11_variance_forecast

DATA = ROOT / "data/processed"
PANEL = DATA / "stage2_multivariate.csv"
COMBINED = DATA / "evaluation_forecasts_with_stage1.csv"
BASELINES = DATA / "stage1_on_stage3_origins_forecasts.csv"
HORIZON = 5


def main() -> None:
    panel = pd.read_csv(PANEL, parse_dates=["date"]).sort_values("date")
    combined = pd.read_csv(COMBINED, parse_dates=["origin_date"])
    baselines = pd.read_csv(BASELINES, parse_dates=["origin_date"])
    if panel["date"].isna().any() or panel["date"].duplicated().any():
        raise ValueError("Panel dates must be nonmissing and unique.")
    if combined["origin_date"].isna().any() or combined.duplicated(
        ["origin_date", "model"]
    ).any():
        raise ValueError("Combined forecasts must have unique nonmissing origins per model.")
    origins = pd.DatetimeIndex(sorted(combined["origin_date"].unique()))
    if len(origins) != 445 or combined["model"].nunique() != 20:
        raise ValueError("Expected 445 origins and 20 models in the saved comparison.")
    panel = panel.set_index("date")
    if not origins.isin(panel.index).all():
        raise ValueError("Some forecast origins are missing from the panel.")
    expected = panel["realized_variance"].reindex(combined["origin_date"])
    if not np.allclose(
        combined["actual"].to_numpy(dtype=float), expected.to_numpy(dtype=float),
        rtol=0, atol=1e-14,
    ):
        raise ValueError("Forecast actuals do not match the panel.")
    baseline_garch = baselines.loc[baselines["model"].eq("garch11")]
    if (
        len(baseline_garch) != len(origins)
        or baseline_garch["origin_date"].isna().any()
        or baseline_garch["origin_date"].duplicated().any()
        or set(baseline_garch["origin_date"]) != set(origins)
    ):
        raise ValueError("Expected one saved GARCH baseline for each origin.")

    rows = []
    for origin in origins:
        history = panel.loc[:origin, "log_return"].dropna().to_numpy(dtype=float)
        try:
            forecast = garch11_variance_forecast(history, horizon=HORIZON)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"GARCH failed at {origin.date()}: {error}") from error
        if not np.isfinite(forecast) or forecast <= 0:
            raise RuntimeError(f"GARCH returned an invalid forecast at {origin.date()}.")
        rows.append({
            "origin_date": origin, "model": "garch11", "forecast": forecast,
            "actual": float(panel.at[origin, "realized_variance"]),
            "converged": True, "error": "",
        })
        if len(rows) % 50 == 0:
            print(f"Fitted {len(rows)}/{len(origins)} GARCH origins.", flush=True)

    results = pd.DataFrame(rows)
    combined = pd.concat(
        [combined.loc[combined["model"].ne("garch11")], results],
        ignore_index=True, sort=False,
    ).sort_values(["origin_date", "model"])
    baselines = pd.concat(
        [baselines.loc[baselines["model"].ne("garch11")], results],
        ignore_index=True, sort=False,
    ).sort_values(["origin_date", "model"])
    counts = combined.groupby("model")["forecast"].agg(["size", "count"])
    if not counts["size"].eq(len(origins)).all() or not counts["count"].eq(len(origins)).all():
        raise ValueError("Not every model has a forecast at each of the 445 origins.")
    if not np.isfinite(combined["forecast"].to_numpy(dtype=float)).all():
        raise ValueError("The aligned comparison contains non-finite forecasts.")
    summary = summary_table(
        combined, benchmark="historical_variance", horizon=HORIZON, top_n=5
    )
    summary["RMSE"] = np.sqrt(summary["MSE"])
    metrics = summary.loc[summary["model"].isin(baselines["model"].unique()), [
        "model", "n", "MSE", "RMSE", "MAE", "QLIKE", "DM_MSE_p_holm",
        "DM_QLIKE_p_holm", "MZ_a", "MZ_b", "MZ_wald_p",
    ]]

    combined.to_csv(COMBINED, index=False)
    combined.to_csv(DATA / "evaluation_forecasts_with_stage1_common_origins.csv", index=False)
    baselines.to_csv(BASELINES, index=False)
    summary.to_csv(DATA / "evaluation_summary_with_stage1.csv", index=False)
    summary.to_csv(DATA / "evaluation_summary_with_stage1_common_origins.csv", index=False)
    metrics.to_csv(DATA / "stage1_on_stage3_origins_metrics.csv", index=False)

    manifest_path = DATA / "stage1_on_stage3_origins_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "common_complete_model_origins": len(origins),
        "garch11_convergence_failures": 0,
        "garch11_refit": "root-mean-square return scaling at each origin",
        "garch11_initial_parameters": [0.05, 0.05, 0.90],
        "garch11_refit_panel_sha256": hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        "garch11_refit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name in ("stage3_var_manifest.json", "stage3_auto_arima_manifest.json"):
        path = DATA / name
        existing_manifest = json.loads(path.read_text(encoding="utf-8"))
        existing_manifest["common_complete_model_origins"] = len(origins)
        existing_manifest["common_comparison_updated_utc"] = manifest["garch11_refit_timestamp_utc"]
        path.write_text(json.dumps(existing_manifest, indent=2) + "\n", encoding="utf-8")

    fig, axis = plt.subplots(figsize=(9.5, 4.0))
    axis.plot(results["origin_date"], results["actual"], label="Actual", color="black", linewidth=1.2)
    axis.plot(results["origin_date"], results["forecast"], label="Forecast", color="#d97941", linewidth=1.0)
    axis.set(title="GARCH(1,1): actual vs forecast", xlabel="Forecast origin (weekly)",
             ylabel="5-session realized variance")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/stage1_on_stage3_origins/garch11.png", dpi=180)
    plt.close(fig)
    print(summary[["model", "n", "MSE", "RMSE", "MAE", "QLIKE",
                   "DM_MSE_p_holm", "DM_QLIKE_p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
