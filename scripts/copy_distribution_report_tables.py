"""Copy gitignored Student-t outputs into the report diagnostics folder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "stage2_yz_distributions"
DEST = ROOT / "report_group2" / "diagnostics"
DISPLAY = {
    "garch": "GARCH(1,1)",
    "egarch": "EGARCH(1,1)",
    "garch_x": "Range GARCH-X",
    "egarch_x": "EGARCH-X",
    "har_x": "HAR-X",
    "hgb": "HGB",
}


def q5_comparison(quintiles: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, distribution), group in quintiles.groupby(
        ["model", "distribution"], sort=False
    ):
        q1 = group.loc[group["forecast_quintile"].eq(1), "error_variance"]
        q5 = group.loc[group["forecast_quintile"].eq(5), "error_variance"]
        metrics = summary.loc[
            summary["model"].eq(model) & summary["distribution"].eq(distribution)
        ].iloc[0]
        rows.append(
            {
                "model": str(model),
                "display_name": DISPLAY.get(str(model), str(model)),
                "distribution": str(distribution),
                "n_common": int(metrics["n_common"]),
                "failure_count": int(metrics["failure_count"]),
                "MAE": float(metrics["MAE"]),
                "RMSE": float(metrics["RMSE"]),
                "QLIKE": float(metrics["QLIKE"]),
                "mean_negative_log_score": float(metrics["mean_negative_log_score"]),
                "coverage_90": float(metrics["coverage_90"]),
                "df_median": (
                    float(metrics["df_median"])
                    if pd.notna(metrics.get("df_median", float("nan")))
                    else float("nan")
                ),
                "q1_variance": float(q1.iloc[0]) if len(q1) else float("nan"),
                "q5_variance": float(q5.iloc[0]) if len(q5) else float("nan"),
                "q5_q1_ratio": (
                    float(q5.iloc[0] / q1.iloc[0])
                    if len(q5) and len(q1) and float(q1.iloc[0]) > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    summary_path = SOURCE / "distribution_summary.csv"
    quintile_path = SOURCE / "forecast_error_quintiles.csv"
    if not summary_path.is_file() or not quintile_path.is_file():
        raise FileNotFoundError(
            f"Student-t outputs not found in {SOURCE}. "
            "Run scripts/compare_volatility_distributions.py first."
        )
    DEST.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    quintiles = pd.read_csv(quintile_path)
    comparison = q5_comparison(quintiles, summary)
    summary.to_csv(DEST / "distribution_summary.csv", index=False)
    quintiles.to_csv(DEST / "forecast_error_quintiles.csv", index=False)
    comparison.to_csv(DEST / "distribution_q5_comparison.csv", index=False)
    manifest = SOURCE / "distribution_manifest.json"
    if manifest.is_file():
        (DEST / "distribution_manifest.json").write_text(
            manifest.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(comparison.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
