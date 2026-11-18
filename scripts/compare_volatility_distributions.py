"""Compare Normal and Student-t forecasts for six Yang-Zhang candidates."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from gold_vol_pipeline.distribution_comparison import (
    evaluate_distribution_comparison,
    save_distribution_comparison,
    summarize_forecast_quintiles,
)
from gold_vol_pipeline.lseg import (
    SELECTED_FORECAST_SOURCES,
    align_selected_forecast_predictors,
)
from gold_vol_pipeline.yang_zhang import (
    add_chronological_split,
    prepare_yang_zhang_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refit GARCH, EGARCH, GARCH-X, EGARCH-X, HAR-X, and HGB under "
            "Normal and standardized Student-t assumptions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/refinitiv/GCc1.csv"),
        help="LSEG GCc1 OHLC CSV.",
    )
    parser.add_argument(
        "--predictor-dir",
        type=Path,
        default=Path("data/refinitiv"),
        help="Directory with DXY and gold IV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stage2_yz_distributions"),
        help="Directory for forecasts, summary, quintiles, and manifest.",
    )
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.window < 2:
        parser.error("--window must be at least 2.")
    if args.workers < 1:
        parser.error("--workers must be positive.")

    prepared = prepare_yang_zhang_dataset(args.input, window=args.window)
    prepared = add_chronological_split(prepared, train_fraction=0.8)
    prepared = align_selected_forecast_predictors(prepared, args.predictor_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = args.output_dir / "GCc1_yang_zhang_volatility.csv"
    prepared.to_csv(prepared_path, index=False)
    valid = prepared.loc[prepared["daily_volatility"].notna()]
    print(
        f"Prepared {len(valid)} Yang-Zhang targets "
        f"({int(valid['split'].eq('train').sum())} train / "
        f"{int(valid['split'].eq('test').sum())} test).",
        flush=True,
    )

    results, summary, manifest = evaluate_distribution_comparison(
        prepared, train_fraction=0.8, workers=args.workers, progress=True
    )
    quintiles = summarize_forecast_quintiles(results)
    manifest.update(
        {
            "source": "LSEG Workspace",
            "input_path": str(args.input),
            "prepared_target_path": str(prepared_path),
            "estimation_window_observations": args.window,
            "annualized": False,
            "holdout_caveat": (
                "The chronological holdout was examined in earlier forecast "
                "comparisons. Scores are descriptive, not a model-selection test."
            ),
            "predictor_inputs": {
                feature: {
                    "path": str(args.predictor_dir / filename),
                    "sha256": hashlib.sha256(
                        (args.predictor_dir / filename).read_bytes()
                    ).hexdigest(),
                }
                for feature, (filename, _, _) in SELECTED_FORECAST_SOURCES.items()
            },
        }
    )
    save_distribution_comparison(
        results, summary, quintiles, manifest, args.output_dir
    )
    print("\nCommon-date Normal versus Student-t comparison:", flush=True)
    columns = [
        "model",
        "distribution",
        "n_common",
        "failure_count",
        "MAE",
        "RMSE",
        "QLIKE",
        "mean_negative_log_score",
        "coverage_90",
        "mean_width_90",
        "df_median",
    ]
    print(
        summary[columns].to_string(
            index=False, float_format=lambda value: f"{value:.8g}"
        ),
        flush=True,
    )
    print(f"\nSaved distribution comparison under {args.output_dir}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
