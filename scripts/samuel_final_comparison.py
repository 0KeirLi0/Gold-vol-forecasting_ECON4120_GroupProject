"""Compare selected Yang-Zhang forecasts and paired DM tests for Samuel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests

from gold_vol_pipeline.volatility_forecasting import (
    SELECTED_PREDICTORS,
    evaluate_daily_volatility,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_FORECASTS = ROOT / "data/processed/stage2_yz_forecasts.csv"
SELECTED_FORECASTS = (
    ROOT / "data/processed/stage2_yz_selected/stage2_yz_forecasts.csv"
)
EGARCHX_FORECASTS = (
    ROOT / "data/processed/stage2_yz_egarchx/stage2_yz_forecasts.csv"
)
BASE_TARGET = ROOT / "data/processed/GCc1_yang_zhang_volatility.csv"
SELECTED_TARGET = (
    ROOT / "data/processed/stage2_yz_selected/GCc1_yang_zhang_volatility.csv"
)
INFORMATION_CRITERIA = (
    ROOT
    / "data/processed/stage2_yz_diagnostics/model_residuals/"
    / "information_criteria.csv"
)
PREVIOUS_METRICS = ROOT / "outputs/samuel/yz_model_comparison.csv"
PREVIOUS_DM = ROOT / "outputs/samuel/yz_dm_results.csv"
OUTPUT = ROOT / "outputs/samuel/final_comparison.csv"

MODEL_IDS = (
    "garch11",
    "egarch",
    "garchx_ohlc_range",
    "egarchx",
    "harx_ohlc",
    "hist_gradient_boosting_ohlc",
)
MODEL_LABELS = {
    "garch11": "GARCH",
    "egarch": "EGARCH",
    "garchx_ohlc_range": "GARCH-X",
    "egarchx": "EGARCH-X",
    "harx_ohlc": "HAR-X",
    "hist_gradient_boosting_ohlc": "hist_gradient_boosting_ohlc",
}
BASE_MODEL_IDS = {"garch11", "egarch"}
SELECTED_MODEL_IDS = {
    "garchx_ohlc_range",
    "harx_ohlc",
    "hist_gradient_boosting_ohlc",
}
DM_BANDWIDTHS = (1, 4)
DM_LOSSES = ("absolute_error", "squared_error")
DM_PAIRS = tuple(
    [("hist_gradient_boosting_ohlc", model) for model in MODEL_IDS if model != "hist_gradient_boosting_ohlc"]
    + [("egarchx", "egarch")]
)


def _validate_forecasts(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "target_date",
        "model",
        "forecast",
        "actual",
        "converged",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{label} must contain {sorted(required)}.")
    result = frame.copy()
    result["target_date"] = pd.to_datetime(result["target_date"], errors="raise")
    result["forecast"] = pd.to_numeric(result["forecast"], errors="raise")
    result["actual"] = pd.to_numeric(result["actual"], errors="raise")
    if result[["target_date", "model"]].isna().any().any():
        raise ValueError(f"{label} has missing target dates or model names.")
    if result.duplicated(["target_date", "model"]).any():
        raise ValueError(f"{label} has duplicate target/model rows.")
    if not np.isfinite(result[["forecast", "actual"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} has non-finite forecasts or actuals.")
    if (result["forecast"] <= 0).any() or (result["actual"] <= 0).any():
        raise ValueError(f"{label} requires positive forecasts and actuals.")
    if not result["converged"].astype(bool).all():
        raise ValueError(f"{label} contains failed or non-converged forecasts.")
    return result.sort_values(["target_date", "model"]).reset_index(drop=True)


def _one_model(
    forecasts: pd.DataFrame, model: str, label: str
) -> pd.DataFrame:
    selected = forecasts.loc[forecasts["model"].eq(model)].copy()
    if selected.empty:
        raise ValueError(f"{label} does not contain model {model!r}.")
    return selected.set_index("target_date")[["forecast", "actual"]].sort_index()


def _egarchx_forecasts() -> tuple[pd.DataFrame, str]:
    if EGARCHX_FORECASTS.exists():
        saved = _validate_forecasts(
            pd.read_csv(EGARCHX_FORECASTS), str(EGARCHX_FORECASTS)
        )
        if "egarchx" in set(saved["model"]):
            return saved.loc[saved["model"].eq("egarchx")], str(EGARCHX_FORECASTS)

    print(
        "EGARCH-X forecast CSV is absent; computing its forecasts in memory "
        "from the selected-predictor target data.",
        flush=True,
    )
    target_data = pd.read_csv(SELECTED_TARGET)
    forecasts, _, _ = evaluate_daily_volatility(
        target_data,
        train_fraction=0.8,
        model_names=("egarchx",),
        workers=4,
        progress=True,
        external_predictors=SELECTED_PREDICTORS,
    )
    return forecasts, f"in-memory evaluation using {SELECTED_TARGET.relative_to(ROOT)}"


def _dm_test(
    loss_difference: np.ndarray, *, bandwidth: int
) -> tuple[float, float]:
    count = len(loss_difference)
    if count < 2:
        raise ValueError("Diebold-Mariano tests require at least two paired forecasts.")
    mean_difference = float(np.mean(loss_difference))
    centered = loss_difference - mean_difference
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, min(bandwidth, count - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1 - lag / (bandwidth + 1)
        long_run_variance += 2 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    if long_run_variance == 0:
        statistic = (
            0.0
            if mean_difference == 0
            else float(np.sign(mean_difference) * np.inf)
        )
    else:
        statistic = mean_difference / np.sqrt(long_run_variance / count)

    horizon = 1
    hln_squared = (
        count + 1 - 2 * horizon + horizon * (horizon - 1) / count
    ) / count
    corrected = statistic * np.sqrt(max(hln_squared, 0.0))
    p_value = (
        float(2 * student_t.sf(abs(corrected), df=count - 1))
        if np.isfinite(corrected)
        else 0.0
    )
    return float(corrected), p_value


def _loss_values(actual: np.ndarray, forecast: np.ndarray, loss: str) -> np.ndarray:
    errors = actual - forecast
    if loss == "absolute_error":
        return np.abs(errors)
    if loss == "squared_error":
        return np.square(errors)
    raise ValueError(f"Unknown loss {loss!r}.")


def _dm_results(
    model_forecasts: dict[str, pd.DataFrame],
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    raw: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    for candidate_name, benchmark_name in DM_PAIRS:
        candidate = model_forecasts[candidate_name]
        benchmark = model_forecasts[benchmark_name]
        if not candidate.index.equals(benchmark.index):
            raise ValueError(
                f"DM pair {candidate_name} vs {benchmark_name} has different dates."
            )
        if not np.allclose(
            candidate["actual"].to_numpy(dtype=float),
            benchmark["actual"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-14,
        ):
            raise ValueError(
                f"DM pair {candidate_name} vs {benchmark_name} has different actuals."
            )
        actual = candidate["actual"].to_numpy(dtype=float)
        candidate_forecast = candidate["forecast"].to_numpy(dtype=float)
        benchmark_forecast = benchmark["forecast"].to_numpy(dtype=float)
        for loss in DM_LOSSES:
            difference = _loss_values(actual, candidate_forecast, loss) - _loss_values(
                actual, benchmark_forecast, loss
            )
            for bandwidth in DM_BANDWIDTHS:
                statistic, p_value = _dm_test(difference, bandwidth=bandwidth)
                raw[(candidate_name, benchmark_name, loss, bandwidth)] = (
                    statistic,
                    p_value,
                )

    adjusted: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for loss in DM_LOSSES:
        for bandwidth in DM_BANDWIDTHS:
            keys = [
                (candidate, benchmark, loss, bandwidth)
                for candidate, benchmark in DM_PAIRS
            ]
            p_values = [raw[key][1] for key in keys]
            holm = multipletests(p_values, method="holm")[1]
            for key, adjusted_p in zip(keys, holm, strict=True):
                adjusted[key] = {
                    "statistic": raw[key][0],
                    "p_raw": raw[key][1],
                    "p_holm": float(adjusted_p),
                }
    return adjusted


def _fit_information_criteria() -> tuple[dict[str, dict[str, float]], str]:
    if INFORMATION_CRITERIA.exists():
        saved = pd.read_csv(INFORMATION_CRITERIA)
        required = {"model", "AIC", "BIC", "method"}
        if not required.issubset(saved.columns):
            raise ValueError(
                f"{INFORMATION_CRITERIA} must contain {sorted(required)}."
            )
        common = saved.loc[saved["model"].isin(("garch11", "egarch", "egarchx"))]
        if (
            set(common["model"]) != {"garch11", "egarch", "egarchx"}
            or not common["method"].eq("arch native likelihood").all()
            or not np.isfinite(common[["AIC", "BIC"]].to_numpy(dtype=float)).all()
        ):
            raise ValueError(
                "Saved information criteria do not provide the common Gaussian "
                "likelihood rows for GARCH, EGARCH, and EGARCH-X."
            )
        return (
            {
                row["model"]: {"AIC": float(row["AIC"]), "BIC": float(row["BIC"])}
                for _, row in common.iterrows()
            },
            str(INFORMATION_CRITERIA),
        )

    baseline = pd.read_csv(BASE_TARGET)
    selected = pd.read_csv(SELECTED_TARGET)
    baseline_train = baseline.loc[baseline["split"].eq("train")].copy()
    selected_train = selected.loc[selected["split"].eq("train")].copy()
    baseline_train = baseline_train.sort_values("date").set_index("date")
    selected_train = selected_train.sort_values("date").set_index("date")
    if not baseline_train.index.equals(selected_train.index):
        raise ValueError("Baseline and selected-predictor training dates differ.")
    values = pd.to_numeric(
        baseline_train["daily_volatility"], errors="raise"
    ).to_numpy(dtype=float)
    selected_values = pd.to_numeric(
        selected_train["daily_volatility"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.allclose(values, selected_values, rtol=0, atol=1e-14):
        raise ValueError(
            "Baseline and selected-predictor training targets differ; "
            "Gaussian information criteria are not comparable."
        )

    specifications = {
        "garch11": {
            "mean": "AR",
            "vol": "GARCH",
            "p": 1,
            "o": 0,
            "q": 1,
        },
        "egarch": {
            "mean": "AR",
            "vol": "EGARCH",
            "p": 1,
            "o": 1,
            "q": 1,
        },
    }
    criteria: dict[str, dict[str, float]] = {}
    for model, options in specifications.items():
        fit = arch_model(
            values * 100,
            lags=1,
            rescale=False,
            **options,
        ).fit(disp="off", show_warning=False)
        if int(getattr(fit, "convergence_flag", 1)) != 0:
            raise RuntimeError(f"{model} training fit did not converge.")
        criteria[model] = {"AIC": float(fit.aic), "BIC": float(fit.bic)}

    features = selected_train.loc[:, SELECTED_PREDICTORS].to_numpy(dtype=float)
    if not np.isfinite(features).all() or (features <= 0).any():
        raise ValueError("EGARCH-X training predictors must be finite and positive.")
    features = features / features.mean(axis=0)
    lagged_features = features.copy()
    lagged_features[1:] = features[:-1]
    egarchx_fit = arch_model(
        selected_values * 100,
        mean="ARX",
        lags=1,
        x=lagged_features,
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        rescale=False,
    ).fit(disp="off", show_warning=False)
    if int(getattr(egarchx_fit, "convergence_flag", 1)) != 0:
        raise RuntimeError("egarchx training fit did not converge.")
    criteria["egarchx"] = {
        "AIC": float(egarchx_fit.aic),
        "BIC": float(egarchx_fit.bic),
    }
    return criteria, "recomputed from the common Gaussian training likelihood"


def _metrics(frame: pd.DataFrame) -> dict[str, float | int | str]:
    errors = frame["actual"].to_numpy(dtype=float) - frame["forecast"].to_numpy(
        dtype=float
    )
    return {
        "n": len(frame),
        "first_target": frame.index.min().date().isoformat(),
        "last_target": frame.index.max().date().isoformat(),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_error_actual_minus_forecast": float(np.mean(errors)),
    }


def _prior_flags(
    model: str,
    metrics: dict[str, float | int | str],
    prior_metrics: pd.DataFrame,
    prior_dm: pd.DataFrame,
    dm_results: dict[tuple[str, str, str, int], dict[str, float]],
) -> str:
    flags: list[str] = []
    previous = prior_metrics.loc[prior_metrics["model"].eq(model)]
    if previous.empty:
        flags.append("no prior Samuel metric row")
    else:
        prior = previous.iloc[0]
        for name in ("MAE", "RMSE", "mean_error_actual_minus_forecast"):
            current_value = float(metrics[name])
            previous_value = float(prior[name])
            if not np.isclose(current_value, previous_value, rtol=0, atol=1e-12):
                flags.append(
                    f"{name}: {current_value:.12g} vs prior {previous_value:.12g}"
                )

    comparisons = (
        [("hist_gradient_boosting_ohlc", model)]
        if model != "hist_gradient_boosting_ohlc"
        else []
    )
    if model == "egarchx":
        comparisons.append(("egarchx", "egarch"))
    for candidate, benchmark in comparisons:
        for loss in DM_LOSSES:
            for bandwidth in DM_BANDWIDTHS:
                previous_test = prior_dm.loc[
                    prior_dm["candidate"].eq(candidate)
                    & prior_dm["benchmark"].eq(benchmark)
                    & prior_dm["loss"].eq(loss)
                    & prior_dm["bandwidth"].eq(bandwidth)
                ]
                if previous_test.empty:
                    continue
                previous_p = float(previous_test.iloc[0]["p_holm"])
                current_p = dm_results[
                    (candidate, benchmark, loss, bandwidth)
                ]["p_holm"]
                if not np.isclose(current_p, previous_p, rtol=0, atol=1e-12):
                    loss_label = "MAE" if loss == "absolute_error" else "RMSE"
                    candidate_label = (
                        "HGB"
                        if candidate == "hist_gradient_boosting_ohlc"
                        else MODEL_LABELS[candidate]
                    )
                    flags.append(
                        f"{candidate_label} vs {MODEL_LABELS[benchmark]} {loss_label} "
                        f"Holm bw{bandwidth}: {current_p:.12g} vs prior "
                        f"{previous_p:.12g}"
                    )
    return "; ".join(flags) if flags else "no numeric differences from matched prior rows"


def main() -> int:
    base = _validate_forecasts(pd.read_csv(BASE_FORECASTS), str(BASE_FORECASTS))
    selected = _validate_forecasts(
        pd.read_csv(SELECTED_FORECASTS), str(SELECTED_FORECASTS)
    )
    egarchx, egarchx_source = _egarchx_forecasts()
    egarchx = _validate_forecasts(egarchx, "EGARCH-X forecasts")

    sources = {
        model: (base if model in BASE_MODEL_IDS else selected)
        for model in MODEL_IDS
        if model != "egarchx"
    }
    sources["egarchx"] = egarchx
    forecast_sources = {
        model: (
            str(BASE_FORECASTS.relative_to(ROOT))
            if model in BASE_MODEL_IDS
            else str(SELECTED_FORECASTS.relative_to(ROOT))
            if model in SELECTED_MODEL_IDS
            else egarchx_source
        )
        for model in MODEL_IDS
    }
    model_forecasts = {
        model: _one_model(
            sources[model],
            model,
            forecast_sources[model],
        )
        for model in MODEL_IDS
    }
    reference_dates = model_forecasts["garch11"].index
    reference_actual = model_forecasts["garch11"]["actual"].to_numpy(dtype=float)
    for model, frame in model_forecasts.items():
        if not frame.index.equals(reference_dates):
            raise ValueError(f"{model} forecasts do not share the common target dates.")
        if not np.allclose(
            frame["actual"].to_numpy(dtype=float),
            reference_actual,
            rtol=0,
            atol=1e-14,
        ):
            raise ValueError(f"{model} forecasts contain different actual targets.")

    dm_results = _dm_results(model_forecasts)
    information_criteria, ic_source = _fit_information_criteria()
    prior_metrics = pd.read_csv(PREVIOUS_METRICS)
    prior_dm = pd.read_csv(PREVIOUS_DM)
    rows: list[dict[str, object]] = []
    for model in MODEL_IDS:
        result: dict[str, object] = {
            "model": MODEL_LABELS[model],
            "model_id": model,
            **_metrics(model_forecasts[model]),
            "forecast_source": forecast_sources[model],
            "AIC": np.nan,
            "BIC": np.nan,
            "AIC_BIC_note": "not reported: likelihood/target differs",
        }
        if model in information_criteria:
            result.update(information_criteria[model])
            result["AIC_BIC_note"] = (
                "training sample only, not comparable across models"
            )
        for loss, loss_label in (
            ("absolute_error", "MAE"),
            ("squared_error", "RMSE"),
        ):
            for bandwidth in DM_BANDWIDTHS:
                column = f"HGB_vs_model_{loss_label}_Holm_p_bw{bandwidth}"
                key = ("hist_gradient_boosting_ohlc", model, loss, bandwidth)
                result[column] = (
                    dm_results[key]["p_holm"]
                    if model != "hist_gradient_boosting_ohlc"
                    else np.nan
                )
                if model == "egarchx":
                    egarchx_key = ("egarchx", "egarch", loss, bandwidth)
                    result[
                        f"EGARCHX_vs_EGARCH_{loss_label}_Holm_p_bw{bandwidth}"
                    ] = dm_results[egarchx_key]["p_holm"]
        result["earlier_outputs_flag"] = _prior_flags(
            model, _metrics(model_forecasts[model]), prior_metrics, prior_dm, dm_results
        )
        rows.append(result)

    table = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(
        "Mean error is actual minus forecast; a positive value means "
        "average under-forecasting."
    )
    print(
        "DM tests use absolute loss (MAE) and squared loss (RMSE), Newey-West "
        "bandwidths 1 and 4, Harvey-Leybourne-Newbold correction, and Holm "
        "adjustment across the six planned comparisons within each loss and "
        "bandwidth."
    )
    print(f"AIC/BIC source: {ic_source}.")
    display_columns = [
        "model",
        "n",
        "MAE",
        "RMSE",
        "mean_error_actual_minus_forecast",
        "AIC",
        "BIC",
        "HGB_vs_model_MAE_Holm_p_bw1",
        "HGB_vs_model_MAE_Holm_p_bw4",
        "HGB_vs_model_RMSE_Holm_p_bw1",
        "HGB_vs_model_RMSE_Holm_p_bw4",
        "EGARCHX_vs_EGARCH_MAE_Holm_p_bw1",
        "EGARCHX_vs_EGARCH_MAE_Holm_p_bw4",
        "EGARCHX_vs_EGARCH_RMSE_Holm_p_bw1",
        "EGARCHX_vs_EGARCH_RMSE_Holm_p_bw4",
    ]
    display_table = table.loc[:, display_columns].rename(
        columns={
            "mean_error_actual_minus_forecast": "Mean error (actual - forecast)",
            "HGB_vs_model_MAE_Holm_p_bw1": "HGB MAE p-Holm NW1",
            "HGB_vs_model_MAE_Holm_p_bw4": "HGB MAE p-Holm NW4",
            "HGB_vs_model_RMSE_Holm_p_bw1": "HGB RMSE p-Holm NW1",
            "HGB_vs_model_RMSE_Holm_p_bw4": "HGB RMSE p-Holm NW4",
            "EGARCHX_vs_EGARCH_MAE_Holm_p_bw1": "EGARCH-X/EGARCH MAE p NW1",
            "EGARCHX_vs_EGARCH_MAE_Holm_p_bw4": "EGARCH-X/EGARCH MAE p NW4",
            "EGARCHX_vs_EGARCH_RMSE_Holm_p_bw1": "EGARCH-X/EGARCH RMSE p NW1",
            "EGARCHX_vs_EGARCH_RMSE_Holm_p_bw4": "EGARCH-X/EGARCH RMSE p NW4",
        }
    )
    print(display_table.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    print(f"\nSaved {OUTPUT.relative_to(ROOT)}")
    changed = table.loc[
        ~table["earlier_outputs_flag"].eq("no numeric differences from matched prior rows"),
        ["model", "earlier_outputs_flag"],
    ]
    if changed.empty:
        print("\nNo numeric differences from matched earlier Samuel outputs.")
    else:
        print("\nEarlier-output differences and new comparisons:")
        for _, row in changed.iterrows():
            print(f"- {row['model']}: {row['earlier_outputs_flag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
