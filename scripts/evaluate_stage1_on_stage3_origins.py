"""Re-evaluate Stage 1 baselines on the completed Stage 3 origin dates."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gold_vol_pipeline.evaluation import summary_table
from gold_vol_pipeline.models import (
    arch1_variance_forecast,
    ar1_squared_return_forecast,
    ewma_variance_forecast,
    expanding_variance_forecast,
    garch11_variance_forecast,
    ma1_squared_return_forecast,
    rolling_variance_forecast,
)

HORIZON = 5
STAGE3_FORECASTS = REPO_ROOT / "data/processed/evaluation_forecasts.csv"
PANEL = REPO_ROOT / "data/processed/stage2_multivariate.csv"
OUTPUT = REPO_ROOT / "data/processed/stage1_on_stage3_origins_forecasts.csv"
COMBINED_OUTPUT = REPO_ROOT / "data/processed/evaluation_forecasts_with_stage1.csv"
SUMMARY_OUTPUT = REPO_ROOT / "data/processed/evaluation_summary_with_stage1.csv"
COMMON_OUTPUT = (
    REPO_ROOT / "data/processed/evaluation_forecasts_with_stage1_common_origins.csv"
)
COMMON_SUMMARY_OUTPUT = (
    REPO_ROOT / "data/processed/evaluation_summary_with_stage1_common_origins.csv"
)
METRICS_OUTPUT = REPO_ROOT / "data/processed/stage1_on_stage3_origins_metrics.csv"
FIGURE_DIR = REPO_ROOT / "reports/figures/stage1_on_stage3_origins"

BASELINE_MODELS = {
    "historical_variance": lambda returns: expanding_variance_forecast(
        returns, horizon=HORIZON
    ),
    "rolling_variance": lambda returns: rolling_variance_forecast(
        returns, horizon=HORIZON, window=20
    ),
    "ewma_fixed": lambda returns: ewma_variance_forecast(
        returns, horizon=HORIZON, decay=0.94
    ),
    "ar1_squared_return": lambda returns: ar1_squared_return_forecast(
        returns, horizon=HORIZON
    ),
    "ma1_squared_return": lambda returns: ma1_squared_return_forecast(
        returns, horizon=HORIZON
    ),
    "arch1": lambda returns: arch1_variance_forecast(
        returns, horizon=HORIZON
    ),
    "garch11": lambda returns: garch11_variance_forecast(
        returns, horizon=HORIZON
    ),
}

DISPLAY_NAMES = {
    "historical_variance": "Expanding historical variance",
    "rolling_variance": "Rolling variance",
    "ewma_fixed": "EWMA, fixed decay 0.94",
    "ar1_squared_return": "AR(1), squared returns",
    "ma1_squared_return": "MA(1), squared returns",
    "arch1": "ARCH(1)",
    "garch11": "GARCH(1,1)",
}


def main() -> None:
    stage3 = pd.read_csv(STAGE3_FORECASTS, parse_dates=["origin_date"])
    panel = pd.read_csv(PANEL, parse_dates=["date", "target_end_date"])
    origins = pd.DatetimeIndex(sorted(stage3["origin_date"].drop_duplicates()))
    if len(origins) != 445:
        raise ValueError(f"Expected 445 Stage 3 origins, found {len(origins)}.")
    panel = panel.sort_values("date").reset_index(drop=True)
    if not origins.isin(panel["date"]).all():
        raise ValueError("Some Stage 3 origins are missing from the prepared panel.")

    panel_by_date = panel.set_index("date")
    stage3_actuals = (
        stage3.drop_duplicates("origin_date")
        .set_index("origin_date")["actual"]
        .sort_index()
    )
    expected_actuals = panel_by_date.loc[origins, "realized_variance"].astype(float)
    expected_actuals.index = origins
    if not np.allclose(
        stage3_actuals.loc[origins].to_numpy(),
        expected_actuals.to_numpy(),
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("Stage 3 actuals do not match the prepared panel.")

    rows = []
    for model_name, model in BASELINE_MODELS.items():
        model_rows = []
        failures = 0
        for origin in origins:
            history = panel.loc[panel["date"] <= origin, "log_return"].dropna()
            actual = float(panel_by_date.at[origin, "realized_variance"])
            try:
                forecast = float(model(history.to_numpy(dtype=float)))
                if not np.isfinite(forecast) or forecast < 0:
                    raise RuntimeError("Model returned an invalid forecast.")
                error = ""
                converged = True
            except (ValueError, RuntimeError) as exception:
                forecast = np.nan
                error = str(exception)
                converged = False
                failures += 1
            model_rows.append(
                {
                    "origin_date": origin,
                    "model": model_name,
                    "forecast": forecast,
                    "actual": actual,
                    "converged": converged,
                    "error": error,
                }
            )
        rows.extend(model_rows)
        print(
            f"{model_name}: {len(origins) - failures}/{len(origins)} forecasts succeeded.",
            flush=True,
        )

    results = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    stage3["origin_date"] = pd.to_datetime(stage3["origin_date"])
    stage1_stage3 = stage3.loc[
        ~stage3["model"].isin(BASELINE_MODELS)
    ].copy()
    stage1_stage3 = pd.concat(
        [stage1_stage3, results], ignore_index=True
    ).sort_values(["origin_date", "model"])
    stage1_stage3.to_csv(COMBINED_OUTPUT, index=False)

    model_origins = [
        set(
            group.loc[group["forecast"].notna(), "origin_date"]
        )
        for _, group in stage1_stage3.groupby("model", sort=False)
    ]
    common_origins = set.intersection(*model_origins)
    common_results = stage1_stage3.loc[
        stage1_stage3["origin_date"].isin(common_origins)
        & stage1_stage3["forecast"].notna()
    ].copy()
    if not common_origins:
        raise ValueError("No complete common-origin sample remains for comparison.")
    counts = common_results.groupby("model").size()
    if counts.nunique() != 1 or counts.iloc[0] != len(common_origins):
        raise ValueError("The final model-comparison sample is not fully balanced.")
    common_results.to_csv(COMMON_OUTPUT, index=False)
    manifest = {
        "evaluation": "fresh Stage 1 baseline fits on exact Stage 3 forecast dates",
        "horizon": HORIZON,
        "origin_count": len(origins),
        "common_complete_model_origins": len(common_origins),
        "garch11_convergence_failures": int(
            results.loc[results["model"].eq("garch11"), "forecast"].isna().sum()
        ),
        "origin_range": {
            "start": origins.min().date().isoformat(),
            "end": origins.max().date().isoformat(),
        },
        "common_origin_range": {
            "start": min(common_origins).date().isoformat(),
            "end": max(common_origins).date().isoformat(),
        },
        "models": list(BASELINE_MODELS),
        "ewma_decay": 0.94,
        "rolling_window": 20,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.with_name("stage1_on_stage3_origins_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary = summary_table(
        stage1_stage3,
        benchmark="historical_variance",
        horizon=HORIZON,
        top_n=5,
    )
    summary["RMSE"] = np.sqrt(summary["MSE"])
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    common_summary = summary_table(
        common_results,
        benchmark="historical_variance",
        horizon=HORIZON,
        top_n=5,
    )
    common_summary["RMSE"] = np.sqrt(common_summary["MSE"])
    common_summary.to_csv(COMMON_SUMMARY_OUTPUT, index=False)
    baseline_summary = common_summary.loc[
        common_summary["model"].isin(BASELINE_MODELS)
    ].copy()
    baseline_summary = baseline_summary[
        [
            "model",
            "n",
            "MSE",
            "RMSE",
            "MAE",
            "QLIKE",
            "DM_MSE_p_holm",
            "DM_QLIKE_p_holm",
            "MZ_a",
            "MZ_b",
            "MZ_wald_p",
        ]
    ]
    baseline_summary.to_csv(METRICS_OUTPUT, index=False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, display_name in DISPLAY_NAMES.items():
        model_rows = results.loc[results["model"].eq(model_name)].sort_values(
            "origin_date"
        )
        figure, axis = plt.subplots(figsize=(9.5, 4.0))
        axis.plot(
            model_rows["origin_date"],
            model_rows["actual"],
            label="Actual",
            color="black",
            linewidth=1.2,
        )
        axis.plot(
            model_rows["origin_date"],
            model_rows["forecast"],
            label="Forecast",
            color="#d97941",
            linewidth=1.0,
        )
        axis.set_title(f"{display_name}: actual vs forecast")
        axis.set_xlabel("Forecast origin (weekly)")
        axis.set_ylabel("5-session realized variance")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(FIGURE_DIR / f"{model_name}.png", dpi=180)
        plt.close(figure)

    print(f"\nWrote forecasts: {OUTPUT}")
    print(f"Wrote aligned combined comparison: {COMBINED_OUTPUT}")
    print(f"Wrote combined model summary: {SUMMARY_OUTPUT}")
    print(f"Wrote common-origin model comparison: {COMMON_SUMMARY_OUTPUT}")
    print(f"Common complete origin count: {len(common_origins)}")
    print(f"Wrote Stage 1 metrics: {METRICS_OUTPUT}")
    print(f"Wrote appendix charts: {FIGURE_DIR}")
    print("\nStage 1 baselines on Stage 3 origins:")
    print(
        baseline_summary[
            ["model", "n", "MSE", "RMSE", "MAE", "QLIKE", "DM_MSE_p_holm"]
        ].to_string(index=False)
    )
    print("\nCombined Stage 1/Stage 3 comparison:")
    print(
        common_summary[
            ["model", "n", "MSE", "RMSE", "MAE", "QLIKE", "DM_MSE_p_holm"]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
