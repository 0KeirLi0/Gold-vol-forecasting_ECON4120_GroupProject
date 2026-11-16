"""Generate common-holdout forecast-error diagnostics for Stage 2 models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gold_vol_pipeline.forecast_diagnostics import (
    build_forecast_error_diagnostics,
    build_forecast_quantile_diagnostics,
    build_regime_and_extreme_diagnostics,
    create_forecast_error_figures,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/stage2_yz_selected"),
        help="Directory containing stage2_yz_forecasts.csv.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("reports/figures/residual_diagnostics"),
        help="Directory for forecast-error plots.",
    )
    parser.add_argument(
        "--acf-lags",
        type=int,
        default=20,
        help="Maximum sample-ACF lag in valid target observations.",
    )
    args = parser.parse_args()
    if args.acf_lags < 1:
        parser.error("--acf-lags must be positive.")

    forecasts_path = args.processed_dir / "stage2_yz_forecasts.csv"
    if not forecasts_path.is_file():
        raise FileNotFoundError(f"Forecast input not found: {forecasts_path}")
    forecasts = pd.read_csv(forecasts_path)
    errors, summary = build_forecast_error_diagnostics(forecasts)
    errors, regimes, extremes, regime_cutoffs = build_regime_and_extreme_diagnostics(
        errors
    )
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    errors_path = args.processed_dir / "forecast_errors.csv"
    summary_path = args.processed_dir / "forecast_error_diagnostics.csv"
    regimes_path = args.processed_dir / "forecast_error_regimes.csv"
    extremes_path = args.processed_dir / "forecast_error_extremes.csv"
    quantiles_path = args.processed_dir / "forecast_error_forecast_quantiles.csv"
    manifest_path = args.processed_dir / "forecast_error_manifest.json"
    errors, quantiles = build_forecast_quantile_diagnostics(errors)
    errors.to_csv(errors_path, index=False)
    summary.to_csv(summary_path, index=False)
    regimes.to_csv(regimes_path, index=False)
    extremes.to_csv(extremes_path, index=False)
    quantiles.to_csv(quantiles_path, index=False)
    figures = create_forecast_error_figures(
        errors,
        str(args.figure_dir),
        acf_lags=args.acf_lags,
        regime_summary=regimes,
        forecast_quantile_summary=quantiles,
    )
    manifest = {
        "input_path": str(forecasts_path),
        "input_sha256": hashlib.sha256(forecasts_path.read_bytes()).hexdigest(),
        "error_definition": "actual minus forecast; positive is under-forecasting",
        "sample": "successful forecasts on target dates shared by every model",
        "common_target_count": int(errors["target_date"].nunique()),
        "models": sorted(errors["model"].unique().tolist()),
        "regime_method": "terciles of realized holdout target; retrospective diagnostic only",
        "regime_cutoffs": regime_cutoffs,
        "forecast_quantile_method": (
            "five equal-count groups formed separately within each model by sorting "
            "forecast values ascending; tied forecasts are ordered by target date"
        ),
        "ljung_box_lags": [5, 10, 20],
        "acf_lags": args.acf_lags,
        "inference_note": (
            "Ljung-Box p-values are exploratory because overlapping targets "
            "induce dependence."
        ),
        "outputs": {
            "errors": str(errors_path),
            "summary": str(summary_path),
            "regimes": str(regimes_path),
            "extremes": str(extremes_path),
            "forecast_quantiles": str(quantiles_path),
            "figures": figures,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in (
        errors_path,
        summary_path,
        regimes_path,
        extremes_path,
        quantiles_path,
        manifest_path,
        *map(Path, figures),
    ):
        print(f"Saved {path}")
    print(f"Common target dates: {manifest['common_target_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
