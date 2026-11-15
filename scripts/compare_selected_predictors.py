"""Compare joint selected-predictor forecasts with the no-external baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gold_vol_pipeline.volatility_forecasting import EXTERNAL_MODEL_NAMES


def compare_forecasts(baseline: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "origin_date", "target_date", "actual", "forecast", "converged"}
    for frame in (baseline, selected):
        if not required <= set(frame.columns):
            raise ValueError(f"Forecast file is missing columns: {sorted(required - set(frame.columns))}")
        if frame.duplicated(["model", "target_date"]).any():
            raise ValueError("Forecast model and target date must be unique.")
    rows = []
    for model in EXTERNAL_MODEL_NAMES:
        left = baseline.loc[baseline["model"].eq(model)]
        right = selected.loc[selected["model"].eq(model)]
        if left.empty or right.empty:
            raise ValueError(f"Forecasts missing for {model}.")
        paired = left.merge(
            right, on=["model", "target_date"], how="outer",
            suffixes=("_baseline", "_selected"), indicator=True, validate="one_to_one",
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError(f"Forecast target dates differ for {model}.")
        if not paired["origin_date_baseline"].eq(paired["origin_date_selected"]).all():
            raise ValueError(f"Forecast origins differ for {model}.")
        if not np.allclose(paired["actual_baseline"], paired["actual_selected"], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Forecast actuals differ for {model}.")
        ok = (
            paired["converged_baseline"].eq(True) & paired["converged_selected"].eq(True)
            & np.isfinite(paired["forecast_baseline"]) & np.isfinite(paired["forecast_selected"])
        )
        common = paired.loc[ok]
        if common.empty:
            raise ValueError(f"No paired successful forecasts for {model}.")
        actual = common["actual_baseline"].to_numpy(dtype=float)
        errors_baseline = common["forecast_baseline"].to_numpy(dtype=float) - actual
        errors_selected = common["forecast_selected"].to_numpy(dtype=float) - actual
        row = {
            "model": model,
            "paired_n": len(common),
            "baseline_failure_count": int((~paired["converged_baseline"].eq(True)).sum()),
            "selected_failure_count": int((~paired["converged_selected"].eq(True)).sum()),
        }
        for label, errors in (("baseline", errors_baseline), ("selected", errors_selected)):
            row[f"{label}_MAE"] = float(np.mean(np.abs(errors)))
            row[f"{label}_RMSE"] = float(np.sqrt(np.mean(errors**2)))
        row["change_MAE"] = row["selected_MAE"] - row["baseline_MAE"]
        row["change_RMSE"] = row["selected_RMSE"] - row["baseline_RMSE"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("data/processed/stage2_yz_forecasts.csv"))
    parser.add_argument("--selected", type=Path, default=Path("data/processed/stage2_yz_selected/stage2_yz_forecasts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/stage2_yz_selected"))
    args = parser.parse_args()
    results = compare_forecasts(pd.read_csv(args.baseline), pd.read_csv(args.selected))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "joint_predictor_comparison.csv", index=False)
    manifest = {
        "baseline_path": str(args.baseline),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "selected_path": str(args.selected),
        "selected_sha256": hashlib.sha256(args.selected.read_bytes()).hexdigest(),
        "comparison": "paired successful forecast dates per model; selected minus baseline",
    }
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
