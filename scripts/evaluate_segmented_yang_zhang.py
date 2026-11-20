"""Evaluate segmented parameter sets for selected Yang-Zhang models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gold_vol_pipeline.segmented_forecasting import (
    SEGMENTED_MODELS,
    TRIGGERS,
    evaluate_segmented_daily_volatility,
    save_segmented_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate normal/extreme segmented parameter sets for GARCH, "
            "EGARCH, GARCH-X, EGARCH-X, HAR-X, and HGB."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/stage2_yz_selected/GCc1_yang_zhang_volatility.csv"
        ),
        help="Prepared Yang-Zhang CSV with selected predictor columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stage2_yz_segmented"),
        help="Directory for segmented forecasts, summaries, and manifest.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.8,
        help="Training-history quantile defining an extreme volatility state.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SEGMENTED_MODELS,
        default=SEGMENTED_MODELS,
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--triggers",
        nargs="+",
        choices=TRIGGERS,
        default=TRIGGERS,
        help="Segment-selection triggers to evaluate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel forecast-origin workers (default: 4).",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    results, summary, manifest = evaluate_segmented_daily_volatility(
        data,
        model_names=tuple(args.models),
        triggers=tuple(args.triggers),
        threshold_quantile=args.threshold_quantile,
        workers=args.workers,
        progress=True,
    )
    save_segmented_evaluation(
        results,
        summary,
        manifest,
        args.output_dir,
        input_path=args.input,
    )
    print("\nSegmented out-of-sample results:", flush=True)
    print(
        summary[
            [
                "model",
                "trigger",
                "common_n",
                "common_MAE",
                "common_RMSE",
                "common_bias",
                "common_QLIKE",
                "failure_count",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.8g}"),
        flush=True,
    )
    print(f"\nSaved segmented outputs under {args.output_dir}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
