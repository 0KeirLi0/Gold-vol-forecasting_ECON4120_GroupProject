"""Integrated Stage 3 rolling-origin evaluation of the extended model set."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .auto_arima import fit_arima_order, hyndman_khandakar
from .evaluation import summary_table
from .models import egarch_asym_forecast, garch_t_forecast, garchx_forecast, gjr_garch_forecast
from .quantile_garch import fit_quantile_garch
from .regression_models import var_forecast


REFITTED_MODELS = (
    "auto_arima_hk", "var", "gjr_garch", "egarch_asym", "garch_t",
    "garchx_vix", "quantile_garch",
)


def _forecast_origin(task: tuple) -> list[dict]:
    origin, returns, vix, actual, order, horizon = task
    dates = pd.RangeIndex(len(returns))
    series = pd.Series(returns, index=dates)
    predictors = pd.DataFrame({"vix_close": vix}, index=dates)

    def arima() -> float:
        fit = fit_arima_order(returns**2, order)
        daily = np.asarray(fit.get_forecast(steps=horizon).predicted_mean)
        return float(np.maximum(daily, 0).sum())

    models = {
        "auto_arima_hk": arima,
        "var": lambda: var_forecast(series, predictors["vix_close"], horizon=horizon, max_lags=5),
        "gjr_garch": lambda: gjr_garch_forecast(returns, horizon=horizon),
        "egarch_asym": lambda: egarch_asym_forecast(returns, horizon=horizon),
        "garch_t": lambda: garch_t_forecast(returns, horizon=horizon),
        "garchx_vix": lambda: garchx_forecast(series, predictors, horizon=horizon),
        "quantile_garch": lambda: fit_quantile_garch(returns).forecast(horizon=horizon),
    }
    rows = []
    for model, forecast_function in models.items():
        try:
            forecast = float(forecast_function())
            if not np.isfinite(forecast) or forecast <= 0:
                raise RuntimeError("Forecast must be finite and strictly positive.")
            error = ""
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exception:
            forecast = np.nan
            error = f"{type(exception).__name__}: {exception}"
        rows.append({
            "origin_date": origin, "model": model, "forecast": forecast,
            "actual": actual, "converged": not error, "error": error,
        })
    return rows


def evaluate_stage3(
    panel_path: Path, cached_path: Path, output_dir: Path, *, workers: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Refit extended candidates and evaluate all candidates on shared origins."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    panel = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date")
    cached = pd.read_csv(cached_path, parse_dates=["origin_date"])
    if panel["date"].isna().any() or panel["date"].duplicated().any():
        raise ValueError("Panel dates must be nonmissing and unique.")
    if cached["origin_date"].isna().any() or cached.duplicated(["origin_date", "model"]).any():
        raise ValueError("Cached forecasts must have unique nonmissing model/origin pairs.")
    origins = pd.DatetimeIndex(sorted(cached["origin_date"].unique()))
    indexed = panel.set_index("date")
    if not origins.isin(indexed.index).all():
        raise ValueError("Some origins are absent from the panel.")
    expected = indexed["realized_variance"].reindex(cached["origin_date"]).to_numpy(dtype=float)
    if not np.allclose(cached["actual"].to_numpy(dtype=float), expected, rtol=0, atol=1e-14):
        raise ValueError("Cached actuals do not match panel targets.")
    tasks = []
    for origin in origins:
        history = indexed.loc[:origin, ["log_return", "vix_close"]]
        if history.isna().any().any():
            raise ValueError(f"Missing return or VIX in history through {origin.date()}.")
        returns = history["log_return"].to_numpy(dtype=float)
        vix = history["vix_close"].to_numpy(dtype=float)
        tasks.append((origin, returns, vix, float(indexed.at[origin, "realized_variance"])))
    selection = hyndman_khandakar(tasks[0][1] ** 2)
    order = selection.order
    fixed_tasks = [(origin, returns, vix, actual, order, 5) for origin, returns, vix, actual in tasks]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_forecast_origin, fixed_tasks):
            rows.extend(result)
            if len(rows) % (25 * len(REFITTED_MODELS)) == 0:
                print(f"Completed {len(rows) // len(REFITTED_MODELS)}/{len(origins)} origins.", flush=True)
    new = pd.DataFrame(rows)
    retained = cached.loc[~cached["model"].isin(REFITTED_MODELS)].copy()
    combined = pd.concat([retained, new], ignore_index=True, sort=False)
    if combined.duplicated(["origin_date", "model"]).any():
        raise ValueError("Combined forecasts contain duplicate model/origin rows.")
    common = set(origins)
    for _, group in combined.groupby("model"):
        valid = group.loc[np.isfinite(group["forecast"]) & (group["forecast"] > 0)]
        common.intersection_update(valid["origin_date"])
    if len(common) < 10:
        raise ValueError("Fewer than 10 complete common origins remain.")
    results = combined.loc[combined["origin_date"].isin(common)].copy()
    counts = results.groupby("model").size()
    if len(counts) != combined["model"].nunique() or not counts.eq(len(common)).all():
        raise ValueError("The common-origin comparison is not balanced.")
    summary = summary_table(results, benchmark="historical_variance", horizon=5, top_n=5)
    summary["RMSE"] = np.sqrt(summary["MSE"])
    manifest = {
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "cached_forecasts_sha256": hashlib.sha256(cached_path.read_bytes()).hexdigest(),
        "origin_count": len(origins), "common_origin_count": len(common),
        "origin_start": min(origins).date().isoformat(),
        "origin_end": max(origins).date().isoformat(),
        "refitted_models": list(REFITTED_MODELS),
        "cached_models": sorted(retained["model"].unique().tolist()),
        "fixed_arima_order": list(order),
        "arima_selection_origin": tasks[0][0].date().isoformat(),
        "failure_counts": new.groupby("model")["converged"].apply(lambda x: int((~x).sum())).to_dict(),
        "horizon": 5, "egarch_simulations": 1000, "quantile": 0.95,
        "quantile_simulations": 20000, "simulation_seed": 42,
        "garchx_future_vix": "held at the last observed value",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "stage3_integrated_attempts.csv", index=False)
    results.to_csv(output_dir / "stage3_integrated_common.csv", index=False)
    summary.to_csv(output_dir / "stage3_integrated_summary.csv", index=False)
    (output_dir / "stage3_integrated_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return results, summary, manifest
