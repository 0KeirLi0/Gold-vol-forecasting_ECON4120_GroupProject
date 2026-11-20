"""Segmented parameter estimation for Yang-Zhang volatility forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .volatility_forecasting import (
    MIN_LOG_VALUE,
    SELECTED_PREDICTORS,
    _metrics,
    _qlike,
    forecast_daily_volatility,
)

SEGMENTED_MODELS = (
    "garch11",
    "egarch",
    "garchx_ohlc_range",
    "egarchx",
    "harx_ohlc",
    "hist_gradient_boosting_ohlc",
)

TRIGGERS = ("forecast_trigger", "lagged_target_trigger")
SEGMENT_QUANTILE = 0.8
EXTERNAL_SEGMENTED_MODELS = {
    "garchx_ohlc_range",
    "egarchx",
    "harx_ohlc",
}


def _regime_history(
    history: pd.DataFrame, *, threshold: float, extreme: bool
) -> pd.DataFrame:
    values = pd.to_numeric(history["daily_volatility"], errors="raise").to_numpy(
        dtype=float
    )
    lagged = np.concatenate(([np.nan], values[:-1]))
    mask = lagged > threshold if extreme else lagged <= threshold
    selected = history.loc[mask].copy()
    if len(selected) < 30:
        label = "extreme" if extreme else "normal"
        raise ValueError(
            f"Segmented {label} regime has only {len(selected)} observations; "
            "at least 30 are required."
        )
    return selected.reset_index(drop=True)


def segmented_forecast(
    model: str,
    history: pd.DataFrame,
    *,
    trigger: str,
    auto_arima_order: tuple[int, int, int] = (1, 0, 0),
    external_predictors: tuple[str, ...] = (),
    threshold_quantile: float = SEGMENT_QUANTILE,
) -> tuple[float, dict[str, object]]:
    """Forecast with a normal/extreme parameter set selected at the origin.

    The separate parameter sets are estimated on observations whose lagged
    realized volatility is normal or extreme. For ``forecast_trigger``, a
    preliminary full-history forecast selects the set; for
    ``lagged_target_trigger``, the latest observed target selects it.
    """
    if trigger not in TRIGGERS:
        raise ValueError(f"Unknown segmented trigger: {trigger}")
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be strictly between zero and one.")
    if model not in SEGMENTED_MODELS:
        raise ValueError(f"Unsupported segmented model: {model}")

    values = pd.to_numeric(history["daily_volatility"], errors="raise").to_numpy(
        dtype=float
    )
    if len(values) < 100:
        raise ValueError("Segmented estimation requires at least 100 observations.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("daily_volatility must be finite and non-negative.")
    threshold = float(np.quantile(values, threshold_quantile))
    model_predictors = (
        external_predictors if model in EXTERNAL_SEGMENTED_MODELS else ()
    )
    pilot_forecast: float | None = None
    if trigger == "forecast_trigger":
        pilot_forecast = forecast_daily_volatility(
            model,
            history,
            auto_arima_order=auto_arima_order,
            external_predictors=model_predictors,
        )
        trigger_value = pilot_forecast
    else:
        trigger_value = float(values[-1])
    extreme = bool(trigger_value > threshold)
    selected = _regime_history(history, threshold=threshold, extreme=extreme)
    forecast = forecast_daily_volatility(
        model,
        selected,
        auto_arima_order=auto_arima_order,
        external_predictors=model_predictors,
    )
    metadata = {
        "segment": "extreme" if extreme else "normal",
        "threshold": threshold,
        "threshold_quantile": threshold_quantile,
        "trigger_value": float(trigger_value),
        "pilot_forecast": pilot_forecast,
        "parameter_sample_count": len(selected),
    }
    return float(max(forecast, MIN_LOG_VALUE)), metadata


def _segmented_origin(
    task: tuple[
        pd.DataFrame,
        pd.Timestamp,
        pd.Timestamp,
        float,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        float,
    ],
) -> list[dict[str, object]]:
    (
        history,
        origin,
        target_date,
        actual,
        model_names,
        triggers,
        external_predictors,
        threshold_quantile,
    ) = task
    rows: list[dict[str, object]] = []
    for model in model_names:
        for trigger in triggers:
            try:
                forecast, metadata = segmented_forecast(
                    model,
                    history,
                    trigger=trigger,
                    external_predictors=external_predictors,
                    threshold_quantile=threshold_quantile,
                )
                error = ""
            except (
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ) as exception:
                forecast = np.nan
                metadata = {
                    "segment": "",
                    "threshold": np.nan,
                    "threshold_quantile": threshold_quantile,
                    "trigger_value": np.nan,
                    "pilot_forecast": np.nan,
                    "parameter_sample_count": 0,
                }
                error = f"{type(exception).__name__}: {exception}"
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": target_date,
                    "model": model,
                    "trigger": trigger,
                    "forecast": forecast,
                    "actual": actual,
                    "converged": not error,
                    "error": error,
                    "history_count": len(history),
                    **metadata,
                }
            )
    return rows


def evaluate_segmented_daily_volatility(
    data: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    model_names: Sequence[str] = SEGMENTED_MODELS,
    triggers: Sequence[str] = TRIGGERS,
    external_predictors: tuple[str, ...] = SELECTED_PREDICTORS,
    threshold_quantile: float = SEGMENT_QUANTILE,
    workers: int = 1,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Evaluate segmented models on a chronological sequential holdout."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one.")
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("model_names must contain unique model names.")
    unknown_models = sorted(set(model_names) - set(SEGMENTED_MODELS))
    if unknown_models:
        raise ValueError(f"Unknown segmented models: {unknown_models}")
    if not triggers or len(set(triggers)) != len(triggers):
        raise ValueError("triggers must contain unique trigger names.")
    unknown_triggers = sorted(set(triggers) - set(TRIGGERS))
    if unknown_triggers:
        raise ValueError(f"Unknown segmented triggers: {unknown_triggers}")
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be strictly between zero and one.")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    if len(set(external_predictors)) != len(external_predictors):
        raise ValueError("external_predictors must not contain duplicates.")
    required = {
        "date",
        "daily_volatility",
        "close_return",
        "overnight_log_return",
        "open_close_log_return",
        "high_low_log_range",
    }
    required.update(external_predictors)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").dropna(subset=["daily_volatility"]).reset_index(
        drop=True
    )
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("data dates must be nonmissing and unique.")
    targets = pd.to_numeric(frame["daily_volatility"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("daily_volatility must contain finite non-negative values.")
    split_position = int(np.floor(len(frame) * train_fraction))
    if split_position < 100 or split_position >= len(frame):
        raise ValueError("The chronological split leaves too few training or test rows.")

    rows: list[dict[str, object]] = []
    test_count = len(frame) - split_position
    tasks = (
        (
            frame.iloc[:test_position].copy(),
            pd.Timestamp(frame.iloc[test_position - 1]["date"]),
            pd.Timestamp(frame.iloc[test_position]["date"]),
            float(frame.iloc[test_position]["daily_volatility"]),
            tuple(model_names),
            tuple(triggers),
            tuple(external_predictors),
            threshold_quantile,
        )
        for test_position in range(split_position, len(frame))
    )
    if workers == 1:
        completed_origins = map(_segmented_origin, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        completed_origins = executor.map(_segmented_origin, tasks, chunksize=1)
    try:
        for completed, origin_rows in enumerate(completed_origins, start=1):
            rows.extend(origin_rows)
            if progress and (completed % 25 == 0 or completed == test_count):
                print(f"Completed {completed}/{test_count} test dates.", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    results = pd.DataFrame(rows)
    successful = results.loc[
        results["converged"] & np.isfinite(results["forecast"])
    ]
    evaluation_names = [
        (model, trigger) for model in model_names for trigger in triggers
    ]
    common_dates = set(frame.iloc[split_position:]["date"])
    for model, trigger in evaluation_names:
        valid = successful.loc[
            successful["model"].eq(model) & successful["trigger"].eq(trigger),
            "target_date",
        ]
        common_dates.intersection_update(valid)
    common = successful.loc[successful["target_date"].isin(common_dates)]

    summary_rows: list[dict[str, object]] = []
    for model, trigger in evaluation_names:
        own = successful.loc[
            successful["model"].eq(model) & successful["trigger"].eq(trigger)
        ]
        shared = common.loc[
            common["model"].eq(model) & common["trigger"].eq(trigger)
        ]
        own_metrics = _metrics(
            own["actual"].to_numpy(dtype=float),
            own["forecast"].to_numpy(dtype=float),
        ) if not own.empty else (np.nan, np.nan, np.nan)
        shared_metrics = _metrics(
            shared["actual"].to_numpy(dtype=float),
            shared["forecast"].to_numpy(dtype=float),
        ) if not shared.empty else (np.nan, np.nan, np.nan)
        summary_rows.append(
            {
                "model": model,
                "trigger": trigger,
                "n_successful": len(own),
                "failure_count": int(
                    (~results.loc[
                        results["model"].eq(model) & results["trigger"].eq(trigger),
                        "converged",
                    ]).sum()
                ),
                "MAE": own_metrics[0],
                "RMSE": own_metrics[1],
                "bias": own_metrics[2],
                "QLIKE": (
                    _qlike(
                        own["actual"].to_numpy(dtype=float),
                        own["forecast"].to_numpy(dtype=float),
                    )
                    if not own.empty
                    else np.nan
                ),
                "common_n": len(shared),
                "common_MAE": shared_metrics[0],
                "common_RMSE": shared_metrics[1],
                "common_bias": shared_metrics[2],
                "common_QLIKE": (
                    _qlike(
                        shared["actual"].to_numpy(dtype=float),
                        shared["forecast"].to_numpy(dtype=float),
                    )
                    if not shared.empty
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["rank_common_MAE"] = summary["common_MAE"].rank(
        method="min", na_option="bottom"
    ).astype("Int64")
    summary["rank_common_RMSE"] = summary["common_RMSE"].rank(
        method="min", na_option="bottom"
    ).astype("Int64")
    summary = summary.sort_values(
        ["rank_common_MAE", "rank_common_RMSE", "model", "trigger"],
        ignore_index=True,
    )
    manifest = {
        "target": "daily_volatility",
        "estimator": "Yang-Zhang",
        "instrument": "COMEX Gold Futures Generic 1 (GCc1)",
        "target_windows_overlap": True,
        "train_fraction": train_fraction,
        "split_method": "chronological by valid target observations",
        "test_forecast_method": (
            "sequential one-step; each regime parameter set is refit on "
            "prior observations whose lagged target is in that regime"
        ),
        "segmented_models": list(model_names),
        "triggers": list(triggers),
        "external_predictors": list(external_predictors),
        "external_predictor_models": sorted(EXTERNAL_SEGMENTED_MODELS),
        "threshold_quantile": threshold_quantile,
        "workers": workers,
        "extreme_rule": "signal > training-history quantile threshold",
        "forecast_trigger_rule": (
            "preliminary full-history model forecast selects normal/extreme set"
        ),
        "lagged_target_trigger_rule": (
            "latest observed y_(t-1) selects normal/extreme set"
        ),
        "training_observations": split_position,
        "test_observations": test_count,
        "common_test_observations": len(common_dates),
        "test_start": pd.Timestamp(frame.iloc[split_position]["date"]).date().isoformat(),
        "test_end": pd.Timestamp(frame.iloc[-1]["date"]).date().isoformat(),
        "ranking_metrics": ["MAE", "RMSE"],
        "reported_metrics": ["MAE", "RMSE", "bias", "QLIKE"],
        "statistical_significance_tests": False,
        "failures": {
            f"{model}__{trigger}": int(
                (~results.loc[
                    results["model"].eq(model) & results["trigger"].eq(trigger),
                    "converged",
                ]).sum()
            )
            for model, trigger in evaluation_names
        },
    }
    return results, summary, manifest


def save_segmented_evaluation(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: dict[str, object],
    output_dir: str | Path,
    *,
    input_path: str | Path,
) -> None:
    """Save segmented forecasts, metrics, and the reproducibility manifest."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / "segmented_yz_forecasts.csv", index=False)
    summary.to_csv(directory / "segmented_yz_summary.csv", index=False)
    output_manifest = dict(manifest)
    output_manifest["input_path"] = str(Path(input_path))
    output_manifest["input_sha256"] = hashlib.sha256(
        Path(input_path).read_bytes()
    ).hexdigest()
    (directory / "segmented_yz_manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
