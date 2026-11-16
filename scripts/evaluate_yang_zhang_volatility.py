"""Prepare Yang-Zhang daily volatility and compare forecasting models."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from gold_vol_pipeline.lseg import (
    SELECTED_FORECAST_SOURCES,
    align_lseg_vix,
    align_selected_forecast_predictors,
)
from gold_vol_pipeline.volatility_forecasting import (
    EXTERNAL_MODEL_NAMES,
    SELECTED_PREDICTORS,
    evaluate_daily_volatility,
    save_daily_volatility_evaluation,
)
from gold_vol_pipeline.yang_zhang import (
    add_chronological_split,
    prepare_yang_zhang_dataset,
)


def _positive_int_list(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if any(not item.strip() for item in parts):
        raise argparse.ArgumentTypeError(
            "rolling windows must be comma-separated positive integers"
        )
    try:
        windows = tuple(int(item.strip()) for item in parts)
    except ValueError as exception:
        raise argparse.ArgumentTypeError(
            "rolling windows must be comma-separated positive integers"
        ) from exception
    if not windows or any(window < 1 for window in windows) or len(set(windows)) != len(windows):
        raise argparse.ArgumentTypeError(
            "rolling windows must be unique positive integers"
        )
    return windows


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
    parser.add_argument(
        "--predictor-dir", type=Path,
        help="LSEG directory containing the selected DXY and gold IV exports.",
    )
    parser.add_argument(
        "--external-models-only", action="store_true",
        help="Evaluate only the six models that accept the selected predictors.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=_positive_int_list,
        help="Also evaluate comma-separated rolling histories (valid target observations).",
    )
    parser.add_argument(
        "--bias-window",
        type=int,
        help="Also apply additive bias calibration using this many prior test errors.",
    )
    parser.add_argument(
        "--hgb-transform-experiments",
        action="store_true",
        help="Compare HGB log-target smearing and raw-target Gamma-loss variants.",
    )
    args = parser.parse_args()

    if args.vix_input is not None and args.predictor_dir is not None:
        parser.error("--vix-input and --predictor-dir are separate experiments.")
    if args.external_models_only and args.predictor_dir is None:
        parser.error("--external-models-only requires --predictor-dir.")
    if args.bias_window is not None and args.bias_window < 1:
        parser.error("--bias-window must be a positive integer.")
    prepared = prepare_yang_zhang_dataset(args.input, window=args.window)
    prepared = add_chronological_split(prepared, train_fraction=0.8)
    if args.vix_input is not None:
        prepared = align_lseg_vix(prepared, args.vix_input)
    if args.predictor_dir is not None:
        prepared = align_selected_forecast_predictors(prepared, args.predictor_dir)
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

    evaluation_options = {
        "rolling_windows": args.rolling_windows or (),
        "bias_window": args.bias_window,
        "hgb_transform_experiments": args.hgb_transform_experiments,
    }
    if args.external_models_only:
        evaluation_options["model_names"] = EXTERNAL_MODEL_NAMES
    results, summary, manifest = evaluate_daily_volatility(
        prepared, train_fraction=0.8, workers=args.workers, progress=True,
        include_vix=args.vix_input is not None,
        external_predictors=SELECTED_PREDICTORS if args.predictor_dir else (),
        **evaluation_options,
    )
    if args.predictor_dir is not None:
        manifest.update({
            "predictor_availability": "latest value available by origin; source date plus one weekday",
            "predictor_models": list(EXTERNAL_MODEL_NAMES),
            "predictor_inputs": {
                feature: {
                    "path": str(args.predictor_dir / filename),
                    "sha256": hashlib.sha256((args.predictor_dir / filename).read_bytes()).hexdigest(),
                }
                for feature, (filename, _, _) in SELECTED_FORECAST_SOURCES.items()
            },
        })
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
                "common_bias",
                "common_QLIKE",
                "failure_count",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.8g}"),
        flush=True,
    )
    print(f"\nSaved forecasts and summary under {args.output_dir}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
