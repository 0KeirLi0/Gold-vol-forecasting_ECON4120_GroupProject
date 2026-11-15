from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline import stage3_evaluation


def test_integrated_evaluation_uses_history_and_common_successes(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=115)
    panel = pd.DataFrame({
        "date": dates,
        "log_return": np.random.default_rng(4).normal(0, 0.01, len(dates)),
        "vix_close": np.full(len(dates), 20.0),
        "realized_variance": np.linspace(0.001, 0.002, len(dates)),
    })
    panel_path = tmp_path / "panel.csv"
    panel.to_csv(panel_path, index=False)
    origins = dates[-12:]
    cached = pd.DataFrame([
        {"origin_date": date, "model": model, "forecast": 0.001,
         "actual": panel.loc[panel.date.eq(date), "realized_variance"].iloc[0],
         "converged": True}
        for date in origins for model in ("historical_variance", "auto_arima_hk")
    ])
    cached_path = tmp_path / "cached.csv"
    cached.to_csv(cached_path, index=False)
    monkeypatch.setattr(
        stage3_evaluation, "hyndman_khandakar",
        lambda values: SimpleNamespace(order=(1, 0, 1)),
    )
    seen = []

    def fake_fit(task):
        origin, returns, vix, actual, order, horizon = task
        seen.append((origin, len(returns), horizon, order))
        return [
            {"origin_date": origin, "model": model,
             "forecast": np.nan if model == "garch_t" and origin == origins[0] else 0.001,
             "actual": actual, "converged": not (model == "garch_t" and origin == origins[0]),
             "error": "failed" if model == "garch_t" and origin == origins[0] else ""}
            for model in stage3_evaluation.REFITTED_MODELS
        ]

    class InlineExecutor:
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, tasks):
            return map(fake_fit, tasks)

    monkeypatch.setattr(stage3_evaluation, "ProcessPoolExecutor", InlineExecutor)
    results, summary, manifest = stage3_evaluation.evaluate_stage3(
        panel_path, cached_path, tmp_path / "output"
    )
    assert len(seen) == len(origins)
    assert seen[0][1] == len(panel.loc[panel.date <= origins[0]])
    assert seen[-1][1] == len(panel)
    assert all(horizon == 5 and order == (1, 0, 1) for _, _, horizon, order in seen)
    assert manifest["common_origin_count"] == 11
    assert manifest["failure_counts"]["garch_t"] == 1
    assert len(summary) == 1 + len(stage3_evaluation.REFITTED_MODELS)
    assert results.groupby("model").size().eq(11).all()
    attempts = pd.read_csv(tmp_path / "output/stage3_integrated_attempts.csv")
    assert attempts.loc[attempts.model.eq("garch_t"), "error"].notna().sum() == 1


def test_garchx_rejects_bad_vix_and_preserves_return_scale():
    from gold_vol_pipeline import garchx_forecast

    returns = np.random.default_rng(13).normal(0, 0.01, 150)
    vix = pd.DataFrame({"vix_close": np.linspace(15, 30, 150)})
    forecast = garchx_forecast(returns, vix, horizon=5)
    scaled = garchx_forecast(returns * 100, vix, horizon=5)
    assert scaled == pytest.approx(forecast * 10_000, rel=1e-5)
    vix.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="finite positive VIX"):
        garchx_forecast(returns, vix)
