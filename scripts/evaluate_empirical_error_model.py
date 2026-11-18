"""Evaluate past-error calibration and empirical prediction intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gold_vol_pipeline.empirical_error_model import (
    calibrate_forecast_errors,
    summarize_calibration,
)
from gold_vol_pipeline.volatility_forecasting import EXTERNAL_MODEL_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasts", type=Path,
        default=Path("data/processed/stage2_yz_diagnostics/stage2_yz_forecasts.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/stage2_yz_diagnostics/empirical_error_model"),
    )
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--hac-lags", type=int, default=10)
    args = parser.parse_args()

    forecasts = pd.read_csv(args.forecasts)
    calibrated = calibrate_forecast_errors(
        forecasts, models=EXTERNAL_MODEL_NAMES, window=args.window, coverage=args.coverage
    )
    summary, mean_tests, _, metadata = summarize_calibration(
        calibrated, coverage=args.coverage, hac_lags=args.hac_lags
    )
    metadata.update({
        "input_path": str(args.forecasts),
        "input_sha256": hashlib.sha256(args.forecasts.read_bytes()).hexdigest(),
        "calibration_window": args.window,
        "warmup": "first window past forecasts per model, excluded from comparison",
        "availability": "prior target dates must be no later than the current forecast origin",
        "scope": "six expanding-history baseline forecasts; fitted sequentially on prior holdout errors",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(args.output_dir / "calibrated_forecasts.csv", index=False)
    summary.to_csv(args.output_dir / "distribution_summary.csv", index=False)
    mean_tests.to_csv(args.output_dir / "forecast_error_tests.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
