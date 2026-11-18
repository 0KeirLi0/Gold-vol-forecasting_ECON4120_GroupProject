"""Check calendar seasonality in the original LSEG GCc1 Yang-Zhang volatility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gold_vol_pipeline.seasonality import (
    build_seasonality_data,
    create_seasonality_figures,
    run_seasonality_tests,
    summarize_calendar_groups,
)
from gold_vol_pipeline.yang_zhang import prepare_yang_zhang_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/refinitiv/GCc1.csv"),
        help="Original LSEG GCc1 OHLC export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/yang_zhang_seasonality"),
        help="Directory for seasonality CSV and JSON outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("reports/figures/yang_zhang_seasonality"),
        help="Directory for seasonality figures.",
    )
    parser.add_argument(
        "--hac-lags",
        type=int,
        nargs="+",
        default=(5, 10, 20),
        help="HAC lag sensitivities for calendar joint tests.",
    )
    args = parser.parse_args()
    if any(lag < 1 for lag in args.hac_lags):
        parser.error("--hac-lags values must be positive.")

    prepared = prepare_yang_zhang_dataset(args.source, window=2)
    observations = build_seasonality_data(prepared)
    month_summary = summarize_calendar_groups(observations, "month")
    weekday_summary = summarize_calendar_groups(observations, "weekday")
    tests = run_seasonality_tests(observations, args.hac_lags)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = args.output_dir / "observations.csv"
    month_path = args.output_dir / "month_summary.csv"
    weekday_path = args.output_dir / "weekday_summary.csv"
    tests_path = args.output_dir / "calendar_tests.csv"
    manifest_path = args.output_dir / "manifest.json"

    observations.to_csv(observations_path, index=False)
    month_summary.to_csv(month_path, index=False)
    weekday_summary.to_csv(weekday_path, index=False)
    tests.to_csv(tests_path, index=False)
    figures = create_seasonality_figures(
        observations,
        month_summary,
        weekday_summary,
        args.figure_dir,
    )

    manifest = {
        "source_path": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "instrument": "LSEG/Refinitiv GCc1 generic COMEX gold futures",
        "target": "unannualized square root of two-observation Yang-Zhang variance",
        "observation_count": int(len(observations)),
        "start_date": observations["date"].min().strftime("%Y-%m-%d"),
        "end_date": observations["date"].max().strftime("%Y-%m-%d"),
        "missing_ohlc_target_count": int(prepared["daily_volatility"].isna().sum()),
        "raw_source_rows": int(len(prepared)),
        "complete_ohlc_rows": int(prepared["complete_ohlc"].sum()),
        "hac_lags": [int(lag) for lag in args.hac_lags],
        "method_note": (
            "Month and weekday joint Wald tests use OLS fixed effects with "
            "HAC standard errors. Lag-adjusted models add the previous valid "
            "Yang-Zhang volatility target. Adjacent targets overlap and HAC "
            "inference is exploratory."
        ),
        "outputs": {
            "observations": str(observations_path),
            "month_summary": str(month_path),
            "weekday_summary": str(weekday_path),
            "calendar_tests": str(tests_path),
            "figures": figures,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for path in (
        observations_path,
        month_path,
        weekday_path,
        tests_path,
        manifest_path,
        *map(Path, figures.values()),
    ):
        print(f"Saved {path}")
    print(
        f"Valid Yang-Zhang observations: {len(observations)} "
        f"({observations['date'].min().date()} to {observations['date'].max().date()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
