"""Test saved sequential forecast errors for bias and variance scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gold_vol_pipeline.forecast_error_diagnostics import diagnose_forecast_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasts",
        type=Path,
        default=Path("data/processed/stage2_yz_diagnostics/stage2_yz_forecasts.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stage2_yz_diagnostics/error_tests"),
    )
    parser.add_argument("--hac-lags", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    forecasts = pd.read_csv(args.forecasts)
    result, metadata = diagnose_forecast_errors(
        forecasts, hac_lags=args.hac_lags, alpha=args.alpha
    )
    metadata["input_path"] = str(args.forecasts)
    metadata["input_sha256"] = hashlib.sha256(args.forecasts.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "forecast_error_tests.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved diagnostic results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
