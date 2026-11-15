"""Prepare Yang-Zhang daily volatility and compare forecasting models."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from gold_vol_pipeline.lseg import align_lseg_vix
from gold_vol_pipeline.volatility_forecasting import (
    evaluate_daily_volatility,
    save_daily_volatility_evaluation,
)
from gold_vol_pipeline.yang_zhang import (
    add_chronological_split,
    prepare_yang_zhang_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build 2-observation Yang-Zhang daily volatility from GCc1 OHLC "
            "and compare one-step forecasts on a chronological 80/20 split."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/refinitiv/GCc1.csv"),
        help="LSEG GCc1 OHLC CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for prepared data, forecasts, summary, and manifest.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Yang-Zhang estimation window in observed OHLC rows (default: 2).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel forecast-origin workers (default: 4).",
    )
    parser.add_argument(
        "--vix-input",
        type=Path,
        help="LSEG VIX CSV to add using the next-weekday availability proxy.",
    )
    args = parser.parse_args()

    prepared = prepare_yang_zhang_dataset(args.input, window=args.window)
    prepared = add_chronological_split(prepared, train_fraction=0.8)
    if args.vix_input is not None:
        prepared = align_lseg_vix(prepared, args.vix_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = args.output_dir / "GCc1_yang_zhang_volatility.csv"
    prepared.to_csv(prepared_path, index=False)

    valid = prepared.loc[prepared["daily_volatility"].notna()]
    train_count = int(valid["split"].eq("train").sum())
    test_count = int(valid["split"].eq("test").sum())
    complete_ohlc_count = int(prepared["complete_ohlc"].sum())
    print(
        f"Prepared {len(prepared)} source rows: {complete_ohlc_count} complete "
        f"OHLC bars, {len(valid)} Yang-Zhang targets "
        f"({train_count} train / {test_count} test).",
        flush=True,
    )
    print(f"Saved daily-volatility data to {prepared_path}.", flush=True)

    results, summary, manifest = evaluate_daily_volatility(
        prepared, train_fraction=0.8, workers=args.workers, progress=True,
        include_vix=args.vix_input is not None,
    )
    if args.vix_input is not None:
        manifest.update(
            {
                "vix_input_file": str(args.vix_input),
                "vix_input_sha256": hashlib.sha256(args.vix_input.read_bytes()).hexdigest(),
                "vix_availability": "latest close available by origin; source date plus one weekday",
                "vix_models": ["harx_ohlc", "ridge_ohlc", "lasso_ohlc",
                               "hist_gradient_boosting_ohlc", "garchx_ohlc_range",
                               "var_ohlc"],
            }
        )
    manifest.update(
        {
            "source": "LSEG Workspace",
            "instrument": "COMEX Gold Futures Generic 1 (GCc1)",
            "input_file": str(args.input),
            "estimation_window_observations": args.window,
            "target_window_observations": args.window,
            "annualized": False,
            "incomplete_ohlc_rows": int((~prepared["complete_ohlc"]).sum()),
            "complete_ohlc_rows": complete_ohlc_count,
            "prepared_target_path": str(prepared_path),
        }
    )
    save_daily_volatility_evaluation(
        results, summary, manifest, args.output_dir, input_path=args.input
    )
    print("\nOut-of-sample results (shared successful dates):", flush=True)
    print(
        summary[
            [
                "model",
                "common_n",
                "common_MAE",
                "common_RMSE",
                "failure_count",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.8g}"),
        flush=True,
    )
    print(f"\nSaved forecasts and summary under {args.output_dir}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
