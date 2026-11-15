"""Plot observed and selected core-model forecasts for the Group 2 report."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/stage3_integrated_common.csv"
OUTPUT = ROOT / "report_group2/figures/core_forecasts.png"

MODELS = {
    "historical_variance": "Historical variance",
    "ewma_fixed": "EWMA (fixed 0.94)",
    "garch11": "GARCH(1,1)",
    "garchx_vix": "GARCH-X (lagged VIX)",
}


def main() -> None:
    forecasts = pd.read_csv(INPUT, parse_dates=["origin_date"])
    selected = forecasts.loc[forecasts["model"].isin(MODELS)].copy()
    actual_counts = selected.groupby("origin_date")["actual"].nunique()
    if not actual_counts.eq(1).all():
        raise ValueError("Selected models do not share the same actual target per origin.")

    pivot = selected.pivot(index="origin_date", columns="model", values="forecast")
    if set(pivot.columns) != set(MODELS) or pivot.isna().any().any():
        raise ValueError("Selected core models must have complete forecasts on common origins.")
    actual = selected.groupby("origin_date")["actual"].first().reindex(pivot.index)

    colors = {
        "historical_variance": "#7b8794",
        "ewma_fixed": "#d68c45",
        "garch11": "#4c8c8a",
        "garchx_vix": "#c65353",
    }
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    ax.plot(actual.index, actual, color="#252a34", linewidth=1.0, alpha=0.8, label="Observed")
    for model, label in MODELS.items():
        ax.plot(
            pivot.index,
            pivot[model],
            color=colors[model],
            linewidth=1.15,
            label=label,
        )
    ax.set_title("Five-session realized variance: observed and core forecasts")
    ax.set_xlabel("Forecast origin")
    ax.set_ylabel("Realized variance (squared log-return units)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT} ({len(pivot)} common origins)")


if __name__ == "__main__":
    main()
