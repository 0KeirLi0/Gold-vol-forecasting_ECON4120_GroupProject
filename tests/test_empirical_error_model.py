import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline.empirical_error_model import (
    calibrate_forecast_errors,
    summarize_calibration,
)


def _forecast_data(size=50):
    dates = pd.bdate_range("2020-01-01", periods=size + 1)
    forecast = np.linspace(0.01, 0.02, size)
    ratio = 0.1 + 0.05 * np.sin(np.arange(size))
    return pd.DataFrame({
        "origin_date": dates[:-1],
        "target_date": dates[1:],
        "model": "example",
        "actual": forecast * (1 + ratio),
        "forecast": forecast,
        "converged": True,
    })


def test_rolling_calibration_uses_only_previous_errors_and_empirical_quantiles():
    frame = _forecast_data()
    result = calibrate_forecast_errors(frame, models=("example",), window=10)
    first = result.iloc[0]
    ratio = (frame["actual"] - frame["forecast"]) / frame["forecast"]
    history = ratio.iloc[:10]
    expected_forecast = frame.iloc[10]["forecast"] * (1 + history.mean())
    assert first["target_date"] == frame.iloc[10]["target_date"]
    assert first["calibration_n"] == 10
    assert first["forecast"] == pytest.approx(expected_forecast)
    assert first["predicted_error_scale"] == pytest.approx(
        frame.iloc[10]["forecast"] * history.std(ddof=1)
    )
    assert first["interval_lower"] == pytest.approx(
        frame.iloc[10]["forecast"] * (1 + history.quantile(0.05))
    )
    assert first["interval_upper"] == pytest.approx(
        frame.iloc[10]["forecast"] * (1 + history.quantile(0.95))
    )
    assert first["standardized_error"] == pytest.approx(
        (frame.iloc[10]["actual"] - expected_forecast) / first["predicted_error_scale"]
    )

    changed = frame.copy()
    changed.loc[10, "actual"] *= 2
    rerun = calibrate_forecast_errors(changed, models=("example",), window=10)
    assert rerun.iloc[0]["forecast"] == pytest.approx(first["forecast"])
    assert rerun.iloc[0]["interval_upper"] == pytest.approx(first["interval_upper"])
    assert rerun.iloc[1]["forecast"] != pytest.approx(result.iloc[1]["forecast"])


def test_rejects_unavailable_history_failures_and_degenerate_scale():
    frame = _forecast_data()
    changed = frame.copy()
    changed.loc[12, "origin_date"] = changed.loc[10, "origin_date"]
    with pytest.raises(ValueError, match="not available"):
        calibrate_forecast_errors(changed, models=("example",), window=10)
    changed = frame.copy()
    changed.loc[1, "converged"] = False
    with pytest.raises(ValueError, match="successful forecasts"):
        calibrate_forecast_errors(changed, models=("example",), window=10)
    changed = frame.assign(actual=frame["forecast"])
    with pytest.raises(ValueError, match="zero or invalid scale"):
        calibrate_forecast_errors(changed, models=("example",), window=10)


def test_summary_uses_common_dates_and_reports_interval_and_distribution():
    frame = _forecast_data(size=100)
    calibrated = calibrate_forecast_errors(frame, models=("example",), window=20)
    summary, tests, comparison, metadata = summarize_calibration(
        calibrated, coverage=0.9, hac_lags=10
    )
    assert len(calibrated) == 80
    assert len(comparison) == 160
    assert tests["n_common"].eq(80).all()
    assert metadata["common_observations"] == 80
    assert summary.loc[0, "empirical_coverage"] == pytest.approx(
        ((calibrated["actual"] >= calibrated["interval_lower"])
         & (calibrated["actual"] <= calibrated["interval_upper"])).mean()
    )
    assert np.isfinite(summary["standardized_squared_slope_p_holm"]).all()
