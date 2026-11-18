import pandas as pd
import pytest

from gold_vol_pipeline.forecast_diagnostics import (
    build_forecast_error_diagnostics,
    build_regime_and_extreme_diagnostics,
)


def _forecasts():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for model, offset in (("model_a", 0.1), ("model_b", -0.2)):
        for position, date in enumerate(dates):
            actual = 1.0 + position / 100
            error = offset + ((position % 5) - 2) * 0.001
            rows.append(
                {
                    "origin_date": date - pd.offsets.BDay(1),
                    "target_date": date,
                    "model": model,
                    "forecast": actual - error,
                    "actual": actual,
                    "converged": True,
                }
            )
    return pd.DataFrame(rows)


def test_forecast_diagnostics_use_positive_actual_minus_forecast_errors():
    errors, summary = build_forecast_error_diagnostics(_forecasts())

    assert len(errors) == 80
    assert errors.loc[
        errors["model"].eq("model_a"), "forecast_error"
    ].mean() == pytest.approx(0.1)
    assert errors.loc[
        errors["model"].eq("model_b"), "forecast_error"
    ].mean() == pytest.approx(-0.2)
    assert summary["n_common"].eq(40).all()
    assert summary["lb_error_p_5"].between(0, 1).all()


def test_forecast_diagnostics_restrict_to_common_successful_dates():
    forecasts = _forecasts()
    failed = forecasts["model"].eq("model_b") & forecasts["target_date"].eq(
        pd.Timestamp("2024-01-02")
    )
    forecasts.loc[failed, "converged"] = False

    errors, summary = build_forecast_error_diagnostics(forecasts)

    assert errors["target_date"].nunique() == 39
    assert summary["n_common"].eq(39).all()
    assert summary.set_index("model").loc["model_b", "failure_count"] == 1


def test_forecast_diagnostics_reject_different_actuals_on_same_date():
    forecasts = _forecasts()
    inconsistent = forecasts["model"].eq("model_b") & forecasts["target_date"].eq(
        pd.Timestamp("2024-01-01")
    )
    forecasts.loc[inconsistent, "actual"] += 1

    with pytest.raises(ValueError, match="different actuals"):
        build_forecast_error_diagnostics(forecasts)


def test_forecast_diagnostics_reject_duplicate_model_dates():
    forecasts = _forecasts()
    forecasts = pd.concat([forecasts, forecasts.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        build_forecast_error_diagnostics(forecasts)


def test_regime_diagnostics_report_retrospective_terciles_and_largest_misses():
    errors, _ = build_forecast_error_diagnostics(_forecasts())

    classified, regimes, extremes, cutoffs = build_regime_and_extreme_diagnostics(
        errors, top_n=2
    )

    assert set(classified["realized_volatility_regime"].astype(str)) == {
        "low",
        "middle",
        "high",
    }
    assert regimes.groupby("model")["n"].sum().eq(40).all()
    assert extremes.groupby("model").size().eq(2).all()
    assert set(extremes.groupby("model")["absolute_error_rank"].apply(tuple)) == {
        (1, 2)
    }
    assert cutoffs["low_upper_bound"] < cutoffs["middle_upper_bound"]
