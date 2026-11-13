import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline.regression_models import (
    PREDICTOR_COLUMNS,
    fit_har_coefficients,
    har_features,
    har_forecast,
    harx_forecast,
    penalized_forecast,
    var_forecast,
)


def _synthetic_inputs(size=260):
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2020-01-01", periods=size)
    returns = pd.Series(rng.normal(0, 0.01, size), index=index)
    predictors = pd.DataFrame(
        {name: rng.uniform(0.5, 2.0, size) for name in PREDICTOR_COLUMNS},
        index=index,
    )
    return returns, predictors


def test_har_features_align_forward_target_and_target_end_date():
    returns = pd.Series(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
        index=pd.date_range("2025-01-01", periods=6),
    )

    features = har_features(returns, horizon=2)

    assert features.iloc[0]["target"] == pytest.approx(returns.iloc[1:3].pow(2).sum())
    assert features.iloc[0]["target_end_date"] == returns.index[2]
    assert features["target"].tail(2).isna().all()
    assert features["rv_d"].iloc[2] == pytest.approx(
        np.log(returns.iloc[2] ** 2 + 1e-12)
    )


@pytest.mark.parametrize(
    ("returns", "horizon", "message"),
    [
        ([0.01, np.nan], 1, "finite numeric"),
        ([0.01, 0.02], 0, "positive integer"),
        ([0.01, 0.02], True, "positive integer"),
    ],
)
def test_har_features_reject_invalid_input(returns, horizon, message):
    with pytest.raises(ValueError, match=message):
        har_features(returns, horizon=horizon)


def test_har_recovers_known_coefficients_from_simulated_har_regression():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.02, 300))
    features = har_features(returns, horizon=5).dropna(subset=["rv_m"])
    x = features[["rv_d", "rv_w", "rv_m"]].to_numpy()
    expected = np.array([0.2, 0.45, 0.3, 0.15])
    log_target = expected[0] + x @ expected[1:]
    coefficients = fit_har_coefficients(features, np.exp(log_target))

    np.testing.assert_allclose(coefficients.to_numpy(), expected, atol=1e-6)


def test_har_and_harx_forecasts_are_nonnegative():
    returns, predictors = _synthetic_inputs()

    assert har_forecast(returns, horizon=5) >= 0
    assert harx_forecast(returns, predictors, horizon=5) >= 0


def test_penalized_forecast_uses_only_training_window_scaling():
    returns, predictors = _synthetic_inputs(size=180)
    changed = predictors.copy()
    changed.iloc[-1] = changed.iloc[-1] * 1000

    original = penalized_forecast(
        returns, predictors, horizon=5, method="lasso", exog=("vix_close",)
    )
    forecast_only_change = penalized_forecast(
        returns, changed, horizon=5, method="lasso", exog=("vix_close",)
    )

    assert original.alpha == forecast_only_change.alpha
    pd.testing.assert_series_equal(
        original.coefficients, forecast_only_change.coefficients
    )


def test_var_forecast_is_nonnegative():
    returns, predictors = _synthetic_inputs(size=180)

    assert var_forecast(returns, predictors["vix_close"], horizon=5) >= 0


def test_regression_models_validate_predictor_alignment_and_penalty_method():
    returns, predictors = _synthetic_inputs()
    shifted = predictors.set_axis(pd.bdate_range("2021-01-01", periods=len(predictors)))

    with pytest.raises(ValueError, match="same index"):
        harx_forecast(returns, shifted, horizon=5)
    with pytest.raises(ValueError, match="method"):
        penalized_forecast(returns, predictors, horizon=5, method="elasticnet")
    with pytest.raises(ValueError, match="unique"):
        penalized_forecast(
            returns,
            predictors,
            horizon=5,
            exog=("vix_close", "vix_close"),
        )


def test_penalized_forecast_accepts_a_reduced_predictor_set():
    returns, predictors = _synthetic_inputs(size=180)

    forecast = penalized_forecast(
        returns,
        predictors,
        horizon=5,
        method="ridge",
        exog=("dxy_close", "vix_close"),
    )

    assert forecast.forecast >= 0
    assert set(forecast.coefficients.index).issubset(
        {"rv_d", "rv_w", "rv_m", "dxy_close", "vix_close",
         "dxy_close_lag1", "vix_close_lag1"}
    )
