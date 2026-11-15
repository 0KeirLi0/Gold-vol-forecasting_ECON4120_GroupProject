"""Create holdout-shaded forecast figures for the Yang-Zhang report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def create_report_figures(
    processed_dir: Path, output_dir: Path
) -> list[Path]:
    """Plot the target split, all model forecasts, and the five MAE leaders."""
    target_path = processed_dir / "GCc1_yang_zhang_volatility.csv"
    forecasts_path = processed_dir / "stage4_yz_forecasts.csv"
    summary_path = processed_dir / "stage4_yz_summary.csv"
    for path in (target_path, forecasts_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation artifact not found: {path}")

    target = pd.read_csv(target_path, parse_dates=["date"])
    forecasts = pd.read_csv(
        forecasts_path, parse_dates=["origin_date", "target_date"]
    )
    summary = pd.read_csv(summary_path).sort_values(
        ["rank_common_MAE", "rank_common_RMSE", "model"]
    )
    if not {"date", "daily_volatility", "split"} <= set(target.columns):
        raise ValueError(f"{target_path} is missing target or split columns.")
    if not {"target_date", "model", "forecast", "actual"} <= set(forecasts.columns):
        raise ValueError(f"{forecasts_path} is missing forecast columns.")

    test_dates = target.loc[target["split"].eq("test"), "date"].dropna()
    if test_dates.empty:
        raise ValueError("The processed target has no test observations.")
    test_start = test_dates.min()
    test_end = test_dates.max()
    target = target.sort_values("date")
    successful = forecasts.loc[
        forecasts["converged"].eq(True) & forecasts["forecast"].notna()
    ].copy()
    model_names = summary["model"].tolist()
    forecast_models = set(successful["model"].unique())
    missing_models = sorted(set(model_names) - forecast_models)
    if missing_models:
        raise ValueError(f"No successful forecast rows for models: {missing_models}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plot_specs = [
        ("Test-period forecast comparison", model_names[:5], "top5_forecasts.png"),
        ("Forecasts: models 1–4 by MAE rank", model_names[0:4], "forecasts_01.png"),
        ("Forecasts: models 5–8 by MAE rank", model_names[4:8], "forecasts_02.png"),
        ("Forecasts: models 9–12 by MAE rank", model_names[8:12], "forecasts_03.png"),
        ("Forecasts: models 13–16 by MAE rank", model_names[12:16], "forecasts_04.png"),
        ("Forecasts: models 17–20 by MAE rank", model_names[16:20], "forecasts_05.png"),
    ]

    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    axis.plot(
        target["date"],
        target["daily_volatility"],
        color="#34495e",
        linewidth=0.75,
        label="Yang–Zhang daily volatility",
    )
    axis.axvspan(
        test_start,
        test_end,
        color="#f4b183",
        alpha=0.25,
        label="Test period",
        zorder=0,
    )
    axis.axvline(test_start, color="#c65911", linestyle="--", linewidth=1)
    axis.set(
        title="Yang–Zhang target and chronological holdout",
        ylabel="Daily volatility (unannualized)",
        xlabel="Date",
    )
    axis.legend(loc="upper left", frameon=False)
    axis.grid(axis="y", alpha=0.2)
    _format_dates(axis)
    split_path = output_dir / "target_split.png"
    figure.savefig(split_path, dpi=180)
    plt.close(figure)
    paths.append(split_path)

    for title, names, filename in plot_specs:
        if not names:
            continue
        if filename == "top5_forecasts.png":
            figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
            _plot_forecasts(
                axis,
                target,
                successful,
                names,
                test_start,
                test_end,
                show_all_actuals=False,
            )
            axis.set_title("Top five models by holdout MAE")
            axis.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
            figure_path = output_dir / filename
        else:
            figure, axes = plt.subplots(
                2, 2, figsize=(12, 8), sharex=True, constrained_layout=True
            )
            for axis, model in zip(axes.flat, names, strict=True):
                _plot_forecasts(
                    axis,
                    target,
                    successful,
                    [model],
                    test_start,
                    test_end,
                    show_all_actuals=True,
                )
                model_summary = summary.loc[summary["model"].eq(model)].iloc[0]
                axis.set_title(
                    f"{model}\nMAE={model_summary['common_MAE']:.5f}; "
                    f"RMSE={model_summary['common_RMSE']:.5f}",
                    fontsize=9,
                )
            axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
            figure.suptitle(title, fontsize=13)
            figure_path = output_dir / filename
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)
        paths.append(figure_path)

    return paths


def _plot_forecasts(
    axis: plt.Axes,
    target: pd.DataFrame,
    forecasts: pd.DataFrame,
    model_names: list[str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    *,
    show_all_actuals: bool,
) -> None:
    actual = target.loc[target["daily_volatility"].notna()]
    if show_all_actuals:
        axis.plot(
            actual["date"],
            actual["daily_volatility"],
            color="#34495e",
            linewidth=0.65,
            label="Actual target",
        )
    else:
        test_actual = actual.loc[actual["date"].between(test_start, test_end)]
        axis.plot(
            test_actual["date"],
            test_actual["daily_volatility"],
            color="#34495e",
            linewidth=0.9,
            label="Actual target",
        )
        axis.axvspan(test_start, test_end, color="#f4b183", alpha=0.2, zorder=0)

    colors = plt.get_cmap("tab10").colors
    for index, model in enumerate(model_names):
        model_forecasts = forecasts.loc[forecasts["model"].eq(model)].sort_values(
            "target_date"
        )
        axis.plot(
            model_forecasts["target_date"],
            model_forecasts["forecast"],
            color=colors[index % len(colors)],
            linewidth=0.8 if show_all_actuals else 1.0,
            alpha=0.9,
            label=model,
        )
    if show_all_actuals:
        axis.axvspan(
            test_start,
            test_end,
            color="#f4b183",
            alpha=0.22,
            label="Test period",
            zorder=0,
        )
        axis.axvline(test_start, color="#c65911", linestyle="--", linewidth=0.8)
    axis.set_ylabel("Volatility")
    axis.set_xlabel("Date")
    axis.grid(axis="y", alpha=0.2)
    _format_dates(axis)


def _format_dates(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator(base=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the prepared target and evaluation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures/stage4_yz"),
        help="Directory for generated report figures.",
    )
    args = parser.parse_args()
    paths = create_report_figures(args.processed_dir, args.output_dir)
    for path in paths:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
