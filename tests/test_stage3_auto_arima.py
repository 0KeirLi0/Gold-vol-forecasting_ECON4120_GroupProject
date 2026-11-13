from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_stage3_auto_arima as evaluation


def test_origin_uses_fixed_order_on_squared_history_and_sums_five_forecasts(monkeypatch):
    seen = []

    def fit(series, order):
        seen.append((series, order))
        return SimpleNamespace(
            get_forecast=lambda steps: SimpleNamespace(
                predicted_mean=np.arange(1, steps + 1, dtype=float)
            ),
        )

    monkeypatch.setattr(evaluation, "fit_arima_order", fit)
    result = evaluation.forecast_origin(
        (pd.Timestamp("2020-01-02"), np.array([-0.1, 0.2]), 0.5, (1, 0, 1))
    )
    np.testing.assert_allclose(seen[0][0], [0.01, 0.04])
    assert seen[0][1] == (1, 0, 1)
    assert result["forecast"] == 15
    assert result["order"] == "(1, 0, 1)"
    assert result["converged"]


def test_failed_fit_is_explicit(monkeypatch):
    def fail(_series, _order):
        raise RuntimeError("no candidate converged")

    monkeypatch.setattr(evaluation, "fit_arima_order", fail)
    result = evaluation.forecast_origin(
        (pd.Timestamp("2020-01-02"), np.array([0.1]), 0.5, (1, 0, 1))
    )
    assert np.isnan(result["forecast"])
    assert not result["converged"]
    assert result["error"] == "no candidate converged"


def test_tasks_exclude_future_returns_and_validate_every_actual():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    panel = pd.DataFrame(
        {"date": dates, "log_return": [0.1, 0.2, 9.0], "realized_variance": [0.4, 0.5, 0.6]}
    )
    forecasts = pd.DataFrame(
        {"origin_date": [dates[1], dates[1]], "model": ["a", "b"],
         "forecast": [0.3, 0.4], "actual": [0.5, 0.5]}
    )
    tasks = evaluation.prepare_tasks(panel, forecasts)
    np.testing.assert_array_equal(tasks[0][1], [0.1, 0.2])
    forecasts.loc[1, "actual"] = 0.6
    with pytest.raises(ValueError, match="actuals do not match"):
        evaluation.prepare_tasks(panel, forecasts)


def test_common_sample_drops_failed_origin_for_every_model():
    dates = pd.date_range("2020-01-01", periods=5)
    forecasts = pd.DataFrame(
        {"origin_date": dates, "model": "historical_variance",
         "forecast": [0.2, 0.3, 0.4, 0.5, 0.6],
         "actual": [0.3, 0.4, 0.2, 0.6, 0.5]}
    )
    results = forecasts.assign(model=evaluation.MODEL, forecast=[0.1, 0.2, np.nan, 0.4, 0.5])
    combined, common, _, summary = evaluation.make_comparison(forecasts, results)
    assert len(combined) == 10
    assert len(common) == 8
    assert dates[2] not in common["origin_date"].values
    assert set(summary["n"]) == {4}
