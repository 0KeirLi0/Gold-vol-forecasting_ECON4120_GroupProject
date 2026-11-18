"""Summarize GCc1 Yang-Zhang volatility by calendar month."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gold_vol_pipeline.yang_zhang_calendar import summarize_yang_zhang_calendar


def _plot_calendar_summary(
    month_summary: pd.DataFrame,
    calendar_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    all_months = month_summary.loc[month_summary["sample"].eq("all")]
    all_calendar = calendar_summary.loc[calendar_summary["sample"].eq("all")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].bar(
        all_months["month"],
        all_months["mean_daily_volatility"],
        color="#b8860b",
        alpha=0.85,
        label="Mean",
    )
    axes[0].plot(
        all_months["month"],
        all_months["median_daily_volatility"],
        color="#263238",
        marker="o",
        linewidth=1.5,
        label="Median",
    )
    axes[0].set_xticks(range(1, 13), all_months["month_name"])
    axes[0].set_ylabel("Unannualized daily volatility")
    axes[0].set_title("Yang-Zhang volatility by calendar month of year")
    axes[0].legend(frameon=False)

    all_calendar = all_calendar.assign(
        calendar_month=pd.to_datetime(
            all_calendar["year"].astype(str)
            + "-"
            + all_calendar["month"].astype(str).str.zfill(2)
            + "-01"
        )
    )
    axes[1].plot(
        all_calendar["calendar_month"],
        all_calendar["mean_daily_volatility"],
        color="#b8860b",
        linewidth=1.3,
    )
    axes[1].set_ylabel("Mean daily volatility")
    axes[1].set_title("Mean Yang-Zhang volatility by calendar year-month")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "GCc1 calendar summaries are descriptive; no seasonality test is applied",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/stage2_yz_selected/GCc1_yang_zhang_volatility.csv"
        ),
        help="Prepared GCc1 target CSV with date, daily_volatility, and split columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stage2_yz_calendar"),
        help="Directory for calendar summary CSVs, plot, and input-hash manifest.",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Yang-Zhang target input not found: {args.input}")

    targets = pd.read_csv(args.input)
    month_summary, calendar_summary = summarize_yang_zhang_calendar(targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    month_path = args.output_dir / "month_of_year_summary.csv"
    calendar_path = args.output_dir / "calendar_month_summary.csv"
    figure_path = args.output_dir / "calendar_month_summaries.png"
    manifest_path = args.output_dir / "manifest.json"
    month_summary.to_csv(month_path, index=False)
    calendar_summary.to_csv(calendar_path, index=False)
    _plot_calendar_summary(month_summary, calendar_summary, figure_path)

    valid = targets.loc[pd.to_numeric(targets["daily_volatility"], errors="coerce").notna()]
    date_values = pd.to_datetime(valid["date"], errors="raise")
    manifest = {
        "analysis": "descriptive GCc1 Yang-Zhang volatility calendar summaries",
        "input_path": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "target": "unannualized square root of the two-observation Yang-Zhang variance",
        "month_of_year_method": (
            "group valid target observations by month number, separately for all, "
            "train, and test samples"
        ),
        "calendar_month_method": (
            "group valid target observations by calendar year and month, separately "
            "for all, train, and test samples"
        ),
        "statistics": [
            "n_targets",
            "n_years",
            "mean_daily_volatility",
            "median_daily_volatility",
            "sd_daily_volatility",
            "q25_daily_volatility",
            "q75_daily_volatility",
            "q90_daily_volatility",
        ],
        "sample": {
            "n_targets": int(len(valid)),
            "start_date": date_values.min().date().isoformat(),
            "end_date": date_values.max().date().isoformat(),
        },
        "interpretation": (
            "descriptive summaries only; not a seasonality hypothesis test or "
            "evidence of a causal calendar effect"
        ),
        "outputs": {
            "month_of_year_summary": str(month_path),
            "calendar_month_summary": str(calendar_path),
            "figure": str(figure_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(month_summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved calendar analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
