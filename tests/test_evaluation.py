import json

import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline.evaluation import (
    diebold_mariano,
    forecast_combination,
    mae,
    mincer_zarnowitz,
    mse,
    qlike,
    rolling_origin_evaluation,
    summary_table,
)


def test_loss_functions_return_origin_losses_and_means():
    actual = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    forecast = pd.Series([2.0, 2.0, 1.0], index=actual.index)

    squared, mean_squared = mse(actual, forecast)
    absolute, mean_absolute = mae(actual, forecast)
    equal_qlike, mean_equal_qlike = qlike(actual, actual)

    assert squared.index.equals(actual.index)
    assert mean_squared == pytest.approx(5 / 3)
    assert mean_absolute == pytest.approx(1.0)
    assert mean_equal_qlike == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(equal_qlike, 0.0)


@pytest.mark.parametrize(
    ("actual", "forecast"),
    [([1.0], [1.0, 2.0]), ([1.0, np.nan], [1.0, 2.0])],
)
def test_losses_reject_invalid_inputs(actual, forecast):
    with pytest.raises(ValueError):
        mse(actual, forecast)


def test_qlike_requires_positive_variances():
    with pytest.raises(ValueError, match="strictly positive"):
        qlike([0.0, 1.0], [0.5, 1.0])


def test_dm_is_zero_for_identical_losses_and_sign_favors_better_model():
    identical = np.array([0.5, 0.3, 0.2, 0.4, 0.1])
    same = diebold_mariano(identical, identical, horizon=5)
    better_a = diebold_mariano(np.zeros(50), np.ones(50), horizon=5)

    assert same.statistic == pytest.approx(0.0)
    assert same.p_value == pytest.approx(1.0)
    assert same.mean_loss_difference == pytest.approx(0.0)
    assert better_a.statistic < 0
    assert better_a.mean_loss_difference < 0


def test_mincer_zarnowitz_returns_calibration_coefficients():
    forecast = np.linspace(0.1, 2.0, 100)
    actual = 0.2 + 1.5 * forecast + np.random.default_rng(10).normal(0, 0.02, 100)

    result = mincer_zarnowitz(actual, forecast, horizon=5)

    assert result.intercept == pytest.approx(0.2, abs=0.01)
    assert result.slope == pytest.approx(1.5, abs=0.01)
    assert 0 <= result.wald_p_value <= 1


def test_forecast_combination_is_equal_weighted():
    combined = forecast_combination(
        {"garch11": 1.0, "ewma": 2.0, "har": 3.0}
    )

    assert combined == pytest.approx(2.0)


def _evaluation_data(size=90):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=size)
    returns = rng.normal(0, 0.01, size)
    target_end = dates[5:]
    target = np.empty(size - 5)
    for index in range(size - 5):
        target[index] = np.square(returns[index + 1:index + 6]).sum()
    return pd.DataFrame(
        {
            "date": dates[:-5],
            "log_return": returns[:-5],
            "target_end_date": target_end,
            "realized_variance": target,
        }
    )


def test_rolling_evaluation_purges_targets_and_saves_manifest(tmp_path):
    data = _evaluation_data()
    seen = []

    def checked_model(context):
        assert (context.training["target_end_date"] <= context.origin).all()
        assert (context.training["date"] <= context.origin).all()
        seen.append(context.origin)
        return 0.001

    output = tmp_path / "nested" / "forecasts.csv"
    result = rolling_origin_evaluation(
        data,
        {"checked": checked_model},
        horizon=5,
        min_train=30,
        step=3,
        output_path=output,
    )

    assert len(result) == len(seen)
    assert result["converged"].all()
    assert (result["forecast"] >= 0).all()
    manifest = json.loads(output.with_name("forecasts_manifest.json").read_text())
    assert manifest["horizon"] == 5
    assert manifest["step"] == 3


def test_rolling_evaluation_records_failed_models_on_common_origins():
    data = _evaluation_data()

    def failed_model(context):
        raise RuntimeError("fit failed")

    result = rolling_origin_evaluation(
        data,
        {"ok": lambda context: 0.001, "failed": failed_model},
        horizon=5,
        min_train=30,
        step=5,
    )

    assert set(result["model"]) == {"ok", "failed"}
    assert result.groupby("origin_date")["model"].nunique().eq(2).all()
    assert not result.loc[result.model == "failed", "converged"].any()
    assert result.loc[result.model == "failed", "error"].eq("fit failed").all()


def test_summary_table_reports_adjusted_dm_and_matrix():
    data = _evaluation_data()
    models = {
        "garch11": lambda context: 0.001 + (len(context.training) % 11) * 1e-6,
        "har": lambda context: 0.0008 + (len(context.training) % 13) * 1e-6,
    }
    forecasts = rolling_origin_evaluation(
        data, models, horizon=5, min_train=30, step=1
    )

    summary = summary_table(forecasts, horizon=5)

    assert {"DM_MSE_p_raw", "DM_MSE_p_holm", "DM_QLIKE_p_raw", "MZ_wald_p"}.issubset(
        summary.columns
    )
    assert "dm_pvalue_matrix_qlike" in summary.attrs
    assert set(summary.model) == {"garch11", "har"}
