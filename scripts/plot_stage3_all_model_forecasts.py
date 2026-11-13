"""Plot actuals and forecasts for every model in the aligned Stage 3 run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FORECASTS = REPO_ROOT / "data/processed/evaluation_forecasts_with_stage1.csv"
OUTPUT_DIR = REPO_ROOT / "reports/figures/stage3_all_model_forecasts"

DISPLAY_NAMES = {
    "historical_variance": "Expanding historical variance",
    "rolling_variance": "Rolling variance",
    "ewma_fixed": "EWMA, fixed decay 0.94",
    "ar1_squared_return": "AR(1), squared returns",
    "ma1_squared_return": "MA(1), squared returns",
    "arch1": "ARCH(1)",
    "garch11": "GARCH(1,1)",
    "ewma": "EWMA, estimated decay",
    "har": "HAR",
    "harx_vix": "HARX, VIX",
    "harx_dxy_vix": "HARX, DXY and VIX",
    "harx_full": "HARX, full predictor set",
    "lasso_vix": "Lasso, VIX",
    "lasso_dxy_vix": "Lasso, DXY and VIX",
    "lasso_full": "Lasso, full predictor set",
    "ridge_vix": "Ridge, VIX",
    "ridge_dxy_vix": "Ridge, DXY and VIX",
    "ridge_full": "Ridge, full predictor set",
    "var": "VAR, BIC-selected",
}


def main() -> None:
    forecasts = pd.read_csv(FORECASTS, parse_dates=["origin_date"])
    required = {"origin_date", "model", "forecast", "actual"}
    if not required.issubset(forecasts.columns):
        raise ValueError(f"{FORECASTS} must contain {sorted(required)}.")

    found = set(forecasts["model"].dropna().unique())
    if found != set(DISPLAY_NAMES):
        missing = sorted(set(DISPLAY_NAMES) - found)
        unexpected = sorted(found - set(DISPLAY_NAMES))
        raise ValueError(f"Model mismatch; missing={missing}, unexpected={unexpected}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, display_name in DISPLAY_NAMES.items():
        rows = forecasts.loc[forecasts["model"].eq(model_name)].sort_values(
            "origin_date"
        )
        if rows["origin_date"].nunique() != 445:
            raise ValueError(
                f"{model_name} has {rows['origin_date'].nunique()} origins, expected 445."
            )
        figure, axis = plt.subplots(figsize=(9.5, 4.0))
        axis.plot(
            rows["origin_date"],
            rows["actual"],
            label="Actual",
            color="black",
            linewidth=1.2,
        )
        axis.plot(
            rows["origin_date"],
            rows["forecast"],
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
        figure.savefig(OUTPUT_DIR / f"{model_name}.png", dpi=180)
        plt.close(figure)

    print(f"Wrote {len(DISPLAY_NAMES)} forecast charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
