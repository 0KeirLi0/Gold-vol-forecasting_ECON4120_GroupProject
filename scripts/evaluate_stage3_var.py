"""Fit the Stage 3 VAR at each saved forecast origin and refresh comparisons."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gold_vol_pipeline.evaluation import summary_table
from gold_vol_pipeline.regression_models import var_forecast

HORIZON = 5
MAX_LAGS = 5
PANEL = REPO_ROOT / "data/processed/stage2_multivariate.csv"
COMBINED_OUTPUT = REPO_ROOT / "data/processed/evaluation_forecasts_with_stage1.csv"
COMMON_OUTPUT = (
    REPO_ROOT / "data/processed/evaluation_forecasts_with_stage1_common_origins.csv"
)
SUMMARY_OUTPUT = REPO_ROOT / "data/processed/evaluation_summary_with_stage1.csv"
COMMON_SUMMARY_OUTPUT = (
    REPO_ROOT / "data/processed/evaluation_summary_with_stage1_common_origins.csv"
)
VAR_OUTPUT = REPO_ROOT / "data/processed/stage3_var_forecasts.csv"
MANIFEST_OUTPUT = REPO_ROOT / "data/processed/stage3_var_manifest.json"


def main() -> None:
    forecasts = pd.read_csv(COMBINED_OUTPUT, parse_dates=["origin_date"])
    panel = pd.read_csv(PANEL, parse_dates=["date"]).sort_values("date")
    required_forecasts = {"origin_date", "model", "forecast", "actual"}
    required_panel = {"date", "log_return", "vix_close", "realized_variance"}
    if not required_forecasts.issubset(forecasts.columns):
        raise ValueError(f"{COMBINED_OUTPUT} must contain {sorted(required_forecasts)}.")
    if not required_panel.issubset(panel.columns):
        raise ValueError(f"{PANEL} must contain {sorted(required_panel)}.")
    if panel["date"].duplicated().any():
        raise ValueError(f"{PANEL} contains duplicate dates.")
    if forecasts.duplicated(["origin_date", "model"]).any():
        raise ValueError(f"{COMBINED_OUTPUT} contains duplicate origin/model rows.")

    origins = pd.DatetimeIndex(sorted(forecasts["origin_date"].dropna().unique()))
    if len(origins) != 445:
        raise ValueError(f"Expected 445 Stage 3 origins, found {len(origins)}.")
    panel_by_date = panel.set_index("date")
    if not origins.isin(panel_by_date.index).all():
        raise ValueError("Some Stage 3 forecast origins are missing from the panel.")

    actuals = forecasts.drop_duplicates("origin_date").set_index("origin_date")["actual"]
    expected_actuals = panel_by_date.loc[origins, "realized_variance"].astype(float)
    if not np.allclose(
        actuals.loc[origins].to_numpy(dtype=float),
        expected_actuals.to_numpy(dtype=float),
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("Saved forecast actuals do not match the prepared panel.")

    rows = []
    for origin in origins:
        history = panel.loc[
            panel["date"].le(origin), ["date", "log_return", "vix_close"]
        ].dropna()
        returns = pd.Series(
            history["log_return"].to_numpy(dtype=float),
            index=history["date"],
            name="log_return",
        )
        vix = pd.Series(
            history["vix_close"].to_numpy(dtype=float),
            index=history["date"],
            name="vix_close",
        )
        actual = float(panel_by_date.at[origin, "realized_variance"])
        try:
            forecast = var_forecast(
                returns, vix, horizon=HORIZON, max_lags=MAX_LAGS
            )
            if not np.isfinite(forecast) or forecast < 0:
                raise RuntimeError("VAR returned an invalid forecast.")
            error = ""
            converged = True
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exception:
            forecast = np.nan
            error = str(exception)
            converged = False
        rows.append(
            {
                "origin_date": origin,
                "model": "var",
                "forecast": forecast,
                "actual": actual,
                "converged": converged,
                "error": error,
            }
        )
    var_results = pd.DataFrame(rows)
    var_results.to_csv(VAR_OUTPUT, index=False)

    existing = forecasts.loc[forecasts["model"].ne("var")].copy()
    combined = pd.concat([existing, var_results], ignore_index=True, sort=False)
    combined = combined.sort_values(["origin_date", "model"])
    common_by_model = [
        set(group.loc[group["forecast"].notna(), "origin_date"])
        for _, group in combined.groupby("model", sort=False)
    ]
    common_origins = set.intersection(*common_by_model)
    common = combined.loc[
        combined["origin_date"].isin(common_origins) & combined["forecast"].notna()
    ].copy()
    counts = common.groupby("model").size()
    if counts.nunique() != 1 or counts.iloc[0] != len(common_origins):
        raise ValueError("The refreshed comparison is not balanced across models.")

    combined.to_csv(COMBINED_OUTPUT, index=False)
    common.to_csv(COMMON_OUTPUT, index=False)
    summary = summary_table(
        combined, benchmark="historical_variance", horizon=HORIZON, top_n=5
    )
    summary["RMSE"] = np.sqrt(summary["MSE"])
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    common_summary = summary_table(
        common, benchmark="historical_variance", horizon=HORIZON, top_n=5
    )
    common_summary["RMSE"] = np.sqrt(common_summary["MSE"])
    common_summary.to_csv(COMMON_SUMMARY_OUTPUT, index=False)

    manifest = {
        "evaluation": "BIC-selected VAR forecasts at the saved Stage 3 origins",
        "variables": ["log daily squared return", "log VIX close"],
        "horizon": HORIZON,
        "max_lags": MAX_LAGS,
        "lag_selection": "BIC, separately at each forecast origin",
        "origin_count": len(origins),
        "successful_forecasts": int(var_results["forecast"].notna().sum()),
        "failed_forecasts": int(var_results["forecast"].isna().sum()),
        "origin_range": {
            "start": origins.min().date().isoformat(),
            "end": origins.max().date().isoformat(),
        },
        "common_complete_model_origins": len(common_origins),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print("\nRefreshed common-origin comparison:")
    print(
        common_summary[
            ["model", "n", "MSE", "RMSE", "MAE", "QLIKE", "DM_MSE_p_holm"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
