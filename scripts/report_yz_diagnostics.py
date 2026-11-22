"""Build report diagnostics: ADF, ACF/PACF, price/vol plots, sequential forecasts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import acf, pacf

from gold_vol_pipeline.auto_arima import hyndman_khandakar
from gold_vol_pipeline.report_diagnostics import (
    SHORTLIST_MODELS,
    acf_pacf_table,
    dickey_fuller_stationary_alternative,
    load_prepared_target,
    load_shortlist_forecasts,
    shortlist_display_name,
    shortlist_holdout_diagnostics,
    split_series,
)
from gold_vol_pipeline.volatility_forecasting import (
    SELECTED_PREDICTORS,
    fit_conditional_variance,
    _lagged_design,
)

MIN_LOG_VALUE = 1e-12
ACF_LAGS = 30


def _format_dates(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator(base=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _shade_holdout(axis: plt.Axes, test_start: pd.Timestamp, test_end: pd.Timestamp) -> None:
    axis.axvspan(test_start, test_end, color="#f4b183", alpha=0.25, zorder=0, label="Holdout")
    axis.axvline(test_start, color="#c65911", linestyle="--", linewidth=1)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_price(target: pd.DataFrame, path: Path) -> Path:
    test = target.loc[target["split"].eq("test"), "date"]
    figure, axis = plt.subplots(figsize=(11.2, 4.4), constrained_layout=True)
    axis.plot(target["date"], target["close"], color="#34495e", linewidth=0.9)
    _shade_holdout(axis, test.min(), test.max())
    axis.set(
        ylabel="Last trade (USD/oz)",
        xlabel="Date",
        title="GCc1 last-trade futures price (generic COMEX gold continuation)",
    )
    axis.legend(loc="upper left", frameon=False)
    axis.grid(axis="y", alpha=0.2)
    _format_dates(axis)
    return _save(figure, path)


def plot_yang_zhang(target: pd.DataFrame, path: Path) -> Path:
    valid = target.loc[target["daily_volatility"].notna()]
    test = valid.loc[valid["split"].eq("test"), "date"]
    figure, axis = plt.subplots(figsize=(11.2, 4.4), constrained_layout=True)
    axis.plot(
        valid["date"],
        valid["daily_volatility"],
        color="#34495e",
        linewidth=0.75,
        label="Yang–Zhang volatility $y_t$",
    )
    _shade_holdout(axis, test.min(), test.max())
    axis.set(
        ylabel="Unannualised daily volatility",
        xlabel="Date",
        title="Next-session Yang–Zhang volatility of GCc1",
    )
    axis.legend(loc="upper left", frameon=False)
    axis.grid(axis="y", alpha=0.2)
    _format_dates(axis)
    return _save(figure, path)


def plot_acf_pacf(series: pd.Series, title: str, path: Path, nlags: int = ACF_LAGS) -> Path:
    values = series.to_numpy(dtype=float)
    acf_point, acf_interval = acf(values, nlags=nlags, fft=True, alpha=0.05)
    pacf_point, pacf_interval = pacf(values, nlags=nlags, method="ywm", alpha=0.05)
    lags = np.arange(nlags + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), constrained_layout=True)
    for axis, point, interval, label in (
        (axes[0], acf_point, acf_interval, "ACF"),
        (axes[1], pacf_point, pacf_interval, "PACF"),
    ):
        axis.axhline(0.0, color="#7f8c8d", linewidth=0.8)
        axis.vlines(lags[1:], 0.0, point[1:], color="#34495e", linewidth=1.1)
        axis.plot(lags[1:], point[1:], ".", color="#34495e")
        band = 1.96 / np.sqrt(len(values))
        axis.axhline(band, color="#c65911", linestyle="--", linewidth=0.8)
        axis.axhline(-band, color="#c65911", linestyle="--", linewidth=0.8)
        axis.fill_between(
            lags[1:],
            interval[1:, 0],
            interval[1:, 1],
            color="#f4b183",
            alpha=0.25,
            linewidth=0,
        )
        axis.set(xlabel="Lag (trading sessions)", ylabel=label, title=label)
        axis.set_xlim(0.5, nlags + 0.5)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(title, fontsize=12)
    return _save(figure, path)


def plot_sequential_forecasts(
    forecasts: pd.DataFrame, path: Path, models: Sequence[str] = SHORTLIST_MODELS
) -> Path:
    figure, axes = plt.subplots(3, 2, figsize=(11.2, 9.4), sharex=True, constrained_layout=True)
    for axis, model in zip(axes.flat, models):
        frame = forecasts.loc[forecasts["model"].astype(str).eq(model)].sort_values(
            "target_date"
        )
        axis.plot(
            frame["target_date"],
            frame["actual"],
            color="#34495e",
            linewidth=0.8,
            label="Actual $y_t$",
        )
        axis.plot(
            frame["target_date"],
            frame["forecast"],
            color="#d55e00",
            linewidth=0.9,
            alpha=0.9,
            label="One-step forecast",
        )
        mae = float(np.mean(np.abs(frame["actual"] - frame["forecast"])))
        axis.set_title(f"{shortlist_display_name(model)}  (MAE {mae:.5f})", fontsize=10)
        axis.set_ylabel("Volatility")
        axis.grid(axis="y", alpha=0.2)
        _format_dates(axis)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("Target date")
    axes[-1, 1].set_xlabel("Target date")
    figure.suptitle(
        "Sequential one-step holdout forecasts, 2023-06-02 to 2026-11-10",
        fontsize=12,
    )
    return _save(figure, path)


def plot_residuals_time(
    errors: pd.DataFrame, path: Path, models: Sequence[str] = SHORTLIST_MODELS
) -> Path:
    figure, axes = plt.subplots(3, 2, figsize=(11.2, 9.4), sharex=True, constrained_layout=True)
    for axis, model in zip(axes.flat, models):
        frame = errors.loc[errors["model"].astype(str).eq(model)].sort_values(
            "target_date"
        )
        axis.axhline(0.0, color="#7f8c8d", linewidth=0.8)
        axis.plot(
            frame["target_date"],
            frame["forecast_error"],
            color="#34495e",
            linewidth=0.7,
        )
        mean_error = float(frame["forecast_error"].mean())
        axis.axhline(mean_error, color="#d55e00", linestyle="--", linewidth=0.9)
        axis.set_title(
            f"{shortlist_display_name(model)}  (mean $e_t$ {mean_error:+.5f})",
            fontsize=10,
        )
        axis.set_ylabel(r"$y_t-\hat y_{t\mid t-1}$")
        axis.grid(axis="y", alpha=0.2)
        _format_dates(axis)
    axes[-1, 0].set_xlabel("Target date")
    axes[-1, 1].set_xlabel("Target date")
    figure.suptitle(
        "Holdout forecast residuals over time (positive = under-forecast)",
        fontsize=12,
    )
    return _save(figure, path)


def _arch_ic(model: str, values: np.ndarray, *, x: np.ndarray | None = None) -> dict[str, object]:
    vol = "EGARCH" if model in {"egarch", "egarchx"} else "GARCH"
    o = 1 if model in {"egarch", "egarchx"} else 0
    fit = arch_model(
        values * 100,
        mean="ARX" if x is not None else "AR",
        lags=1,
        x=x,
        vol=vol,
        p=1,
        o=o,
        q=1,
        rescale=False,
    ).fit(disp="off", show_warning=False)
    if int(getattr(fit, "convergence_flag", 1)) != 0:
        raise RuntimeError(f"{model} training-origin fit did not converge.")
    return {
        "model": model,
        "n_training_observations": int(len(values)),
        "n_parameters": int(len(fit.params)),
        "log_likelihood": float(fit.loglikelihood),
        "AIC": float(fit.aic),
        "BIC": float(fit.bic),
        "method": "arch native likelihood at the training origin",
        "selection_rule": "course default GARCH(1,1)/EGARCH(1,1); AIC/BIC reported, not used to pick p,q",
    }


def _harx_ic(history: pd.DataFrame) -> dict[str, object]:
    features, targets, _ = _lagged_design(history, include_ohlc=True)
    design = np.column_stack((np.ones(len(features)), features))
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    residuals = targets - design @ coefficients
    variance = float(np.mean(np.square(residuals)))
    log_likelihood = -0.5 * len(residuals) * (np.log(2 * np.pi) + 1 + np.log(variance))
    parameters = design.shape[1] + 1
    return {
        "model": "harx_ohlc",
        "n_training_observations": int(len(residuals)),
        "n_parameters": int(parameters),
        "log_likelihood": float(log_likelihood),
        "AIC": float(2 * parameters - 2 * log_likelihood),
        "BIC": float(parameters * np.log(len(residuals)) - 2 * log_likelihood),
        "method": "Gaussian OLS likelihood on log volatility",
        "selection_rule": "Corsi HAR lags 1/5/22 plus origin-known OHLC; not AIC-selected",
    }


def training_information_criteria(
    baseline: pd.DataFrame, egarchx: pd.DataFrame
) -> pd.DataFrame:
    train = baseline.loc[baseline["split"].eq("train")].copy()
    values = pd.to_numeric(train["daily_volatility"], errors="raise").to_numpy(dtype=float)
    rows = [_arch_ic("garch11", values), _arch_ic("egarch", values)]
    egarchx_train = egarchx.loc[egarchx["split"].eq("train")].copy()
    egarchx_values = pd.to_numeric(
        egarchx_train["daily_volatility"], errors="raise"
    ).to_numpy(dtype=float)
    features = egarchx_train.loc[:, list(SELECTED_PREDICTORS)].to_numpy(dtype=float)
    features = features / features.mean(axis=0)
    lagged = features.copy()
    lagged[1:] = features[:-1]
    rows.append(_arch_ic("egarchx", egarchx_values, x=lagged))
    rows.append(_harx_ic(train))
    range_fit = fit_conditional_variance(
        train, tail_threshold=None, external_predictors=()
    )
    rows.append(
        {
            "model": "garchx_ohlc_range",
            "n_training_observations": int(range_fit.n_observations),
            "n_parameters": int(range_fit.n_parameters),
            "log_likelihood": float(range_fit.log_likelihood),
            "AIC": float(2 * range_fit.n_parameters - 2 * range_fit.log_likelihood),
            "BIC": float(
                range_fit.n_parameters * np.log(range_fit.n_observations)
                - 2 * range_fit.log_likelihood
            ),
            "method": "custom constrained Gaussian quasi-likelihood",
            "selection_rule": "course GARCH-X with OHLC range; p=q=1 not AIC-selected",
        }
    )
    rows.append(
        {
            "model": "hist_gradient_boosting_ohlc",
            "n_training_observations": np.nan,
            "n_parameters": np.nan,
            "log_likelihood": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "method": "not applicable: no specified likelihood",
            "selection_rule": "configured learning_rate=0.05, max_iter=100, max_leaf_nodes=15, seed=42",
        }
    )
    return pd.DataFrame(rows)


def arima_selection(train_log: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = hyndman_khandakar(
        train_log, max_p=3, max_q=3, max_d=2, max_order=4
    )
    candidates = selected.candidates.copy()
    summary = {
        "selected_order": list(selected.order),
        "selected_aicc": float(selected.aicc),
        "differencing_kpss_p_values": list(selected.differencing_p_values),
        "selection_rule": (
            "Hyndman-Khandakar: repeated KPSS chooses d; stepwise AICc chooses p,q "
            "with max_p=max_q=3, max_d=2, max_order=4 on training log y_t"
        ),
    }
    return candidates, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/processed/GCc1_yang_zhang_volatility.csv"),
    )
    parser.add_argument(
        "--baseline-forecasts",
        type=Path,
        default=Path("data/processed/stage2_yz_forecasts.csv"),
    )
    parser.add_argument(
        "--egarchx-forecasts",
        type=Path,
        default=Path("data/processed/stage2_yz_egarchx/stage2_yz_forecasts.csv"),
    )
    parser.add_argument(
        "--egarchx-target",
        type=Path,
        default=Path("data/processed/stage2_yz_egarchx/GCc1_yang_zhang_volatility.csv"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("report_group2/figures"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report_group2/diagnostics"),
    )
    args = parser.parse_args()

    target = load_prepared_target(args.target)
    y = split_series(target, "daily_volatility")
    price = split_series(target, "close", valid_only=True)
    log_y = {
        sample: np.log(np.maximum(series, MIN_LOG_VALUE))
        for sample, series in y.items()
    }
    price_diff = {
        sample: series.diff().dropna() for sample, series in price.items()
    }

    adf_rows = []
    for sample in ("train", "full"):
        adf_rows.append(
            dickey_fuller_stationary_alternative(
                y[sample], name="yang_zhang_volatility", sample=sample
            )
        )
        adf_rows.append(
            dickey_fuller_stationary_alternative(
                log_y[sample], name="log_yang_zhang_volatility", sample=sample
            )
        )
        adf_rows.append(
            dickey_fuller_stationary_alternative(
                price[sample], name="gcc1_last_price", sample=sample
            )
        )
        adf_rows.append(
            dickey_fuller_stationary_alternative(
                price_diff[sample], name="gcc1_last_price_first_difference", sample=sample
            )
        )
    adf = pd.DataFrame(adf_rows)

    acf_tables = [
        acf_pacf_table(y["train"], name="yang_zhang_volatility", sample="train"),
        acf_pacf_table(log_y["train"], name="log_yang_zhang_volatility", sample="train"),
    ]
    acf_all = pd.concat(acf_tables, ignore_index=True)

    forecasts = load_shortlist_forecasts(
        baseline_path=args.baseline_forecasts,
        egarchx_path=args.egarchx_forecasts,
    )
    errors, lb_summary = shortlist_holdout_diagnostics(forecasts)
    errors["display_name"] = errors["model"].astype(str).map(shortlist_display_name)
    lb_summary["display_name"] = lb_summary["model"].map(shortlist_display_name)

    egarchx_target = load_prepared_target(args.egarchx_target)
    ic = training_information_criteria(target, egarchx_target)
    train_log = log_y["train"].to_numpy(dtype=float)
    hk_candidates, hk_summary = arima_selection(train_log)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adf.to_csv(args.output_dir / "adf_tests.csv", index=False)
    acf_all.to_csv(args.output_dir / "acf_pacf.csv", index=False)
    ic.to_csv(args.output_dir / "information_criteria.csv", index=False)
    hk_candidates.to_csv(args.output_dir / "arima_hk_candidates.csv", index=False)
    lb_summary.to_csv(args.output_dir / "holdout_ljung_box.csv", index=False)
    errors.to_csv(args.output_dir / "holdout_forecast_errors.csv", index=False)

    figures = [
        plot_price(target, args.figure_dir / "gcc1_price_date.png"),
        plot_yang_zhang(target, args.figure_dir / "yz_volatility_date.png"),
        plot_acf_pacf(
            y["train"],
            "ACF and PACF of training Yang–Zhang volatility $y_t$",
            args.figure_dir / "yz_acf_pacf.png",
        ),
        plot_acf_pacf(
            log_y["train"],
            r"ACF and PACF of training $\log y_t$",
            args.figure_dir / "log_yz_acf_pacf.png",
        ),
        plot_sequential_forecasts(
            forecasts, args.figure_dir / "shortlist_sequential_forecasts.png"
        ),
        plot_residuals_time(errors, args.figure_dir / "shortlist_residuals_time.png"),
    ]

    manifest = {
        "target": str(args.target),
        "n_valid_yang_zhang": int(y["full"].size),
        "n_train": int(y["train"].size),
        "n_test": int(y["test"].size),
        "adf_alternative": "stationary",
        "adf_autolag": "AIC",
        "adf_regression": "c",
        "acf_lags": ACF_LAGS,
        "shortlist_models": list(SHORTLIST_MODELS),
        "hyndman_khandakar": hk_summary,
        "figures": [str(path) for path in figures],
        "note": (
            "ADF uses statsmodels adfuller with the stationary alternative and "
            "default AIC lag selection. Sequential forecasts are the saved "
            "Stage 2 one-step refits; models are not re-estimated here."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(adf.to_string(index=False))
    print()
    print(ic.to_string(index=False))
    print()
    print("HK selected order:", hk_summary["selected_order"], "AICc", hk_summary["selected_aicc"])
    print("Saved", args.output_dir)
    for path in figures:
        print("Saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
