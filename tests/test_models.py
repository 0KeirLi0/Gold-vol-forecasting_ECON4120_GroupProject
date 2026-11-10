import sys

import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline import (
    arch1_variance_forecast,
    ar1_squared_return_forecast,
    baseline_forecasts,
    ewma_variance_forecast,
    expanding_variance_forecast,
    garch11_variance_forecast,
    ma1_squared_return_forecast,
    main,
    prediction_history,
    rolling_variance_forecast,
)


def test_main_prints_prediction_results(tmp_path, capsys, monkeypatch):
    csv_path = tmp_path / "returns.csv"
    pd.DataFrame({"log_return": [0.01, -0.02, 0.03, 0.02, -0.01]}).to_csv(
        csv_path, index=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "models.py",
            "--input",
            str(csv_path),
            "--horizon",
            "2",
            "--rolling-window",
            "3",
            "--ewma-decay",
            "0.9",
        ],
    )

    result = main()
    captured = capsys.readouterr().out

    assert result == 0
    assert "Forecast horizon: 2" in captured
    assert "historical_variance:" in captured
    assert "garch11:" in captured


def test_prediction_history_uses_full_prior_history_for_recent_origins(tmp_path):
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    csv_path = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(returns)),
            "log_return": returns,
            "realized_variance": np.full(len(returns), 0.001),
        }
    ).to_csv(csv_path, index=False)

    comparison = prediction_history(str(csv_path), horizon=2, sample_size=1)

    assert len(comparison) == 1
    assert comparison.iloc[0]["date"] == "2025-01-05"
    assert comparison.iloc[0]["historical_variance"] == pytest.approx(
        2 * np.mean(returns**2)
    )


def test_historical_and_rolling_variance_forecasts_sum_over_horizon():
    returns = pd.Series([0.01, -0.02, 0.03])

    assert expanding_variance_forecast(returns, horizon=2) == pytest.approx(
        2 * np.mean(np.square(returns))
    )
    assert rolling_variance_forecast(
        returns, horizon=2, window=2
    ) == pytest.approx(2 * np.mean(np.square(returns.iloc[-2:])))


def test_ewma_uses_only_returns_through_forecast_origin():
    returns = [0.01, -0.02, 0.04]
    decay = 0.8
    expected_variance = (
        decay * (decay * returns[0] ** 2 + (1 - decay) * returns[1] ** 2)
        + (1 - decay) * returns[2] ** 2
    )

    assert ewma_variance_forecast(
        returns, horizon=3, decay=decay
    ) == pytest.approx(3 * expected_variance)


def test_all_baseline_forecasts_return_nonnegative_values():
    returns = np.random.default_rng(42).normal(0, 0.01, size=120)

    forecasts = baseline_forecasts(returns, horizon=5, rolling_window=30)

    assert set(forecasts) == {
        "historical_variance",
        "rolling_variance",
        "ewma_variance",
        "ar1_squared_return",
        "ma1_squared_return",
        "arch1",
        "garch11",
    }
    assert all(np.isfinite(value) and value >= 0 for value in forecasts.values())


def test_all_baselines_handle_zero_returns():
    forecasts = baseline_forecasts(np.zeros(40), horizon=5)

    assert all(value == 0 for value in forecasts.values())


@pytest.mark.parametrize(
    "forecast",
    [
        ar1_squared_return_forecast,
        ma1_squared_return_forecast,
        arch1_variance_forecast,
        garch11_variance_forecast,
    ],
)
def test_fitted_baselines_forecast_multiple_periods(forecast):
    returns = np.random.default_rng(7).normal(0, 0.01, size=80)

    value = forecast(returns, horizon=5)

    assert np.isfinite(value)
    assert value >= 0


@pytest.mark.parametrize(
    ("forecast", "returns", "kwargs", "message"),
    [
        (rolling_variance_forecast, [0.01, np.nan], {}, "finite numeric"),
        (rolling_variance_forecast, [0.01, 0.02], {"horizon": 0}, "positive integer"),
        (rolling_variance_forecast, [0.01, 0.02], {"window": 0}, "positive integer"),
        (ewma_variance_forecast, [0.01, 0.02], {"decay": 1}, "strictly between"),
    ],
)
def test_baselines_reject_invalid_input(forecast, returns, kwargs, message):
    with pytest.raises(ValueError, match=message):
        forecast(returns, **kwargs)
