import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

from gold_vol_pipeline.forecast_diagnostics import (
    build_forecast_error_diagnostics,
    build_forecast_quantile_diagnostics,
    build_mean_error_t_tests,
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


def test_forecast_diagnostics_can_select_models_before_common_date_matching():
    forecasts = _forecasts()
    extra_model = forecasts.loc[forecasts["model"].eq("model_a")].copy()
    extra_model["model"] = "model_c"
    forecasts = pd.concat([forecasts, extra_model], ignore_index=True)

    errors, summary = build_forecast_error_diagnostics(
        forecasts, models=("model_a", "model_b")
    )

    assert set(errors["model"]) == {"model_a", "model_b"}
    assert set(summary["model"]) == {"model_a", "model_b"}
    assert summary["n_common"].eq(40).all()


def test_forecast_diagnostics_reject_missing_or_duplicate_selected_models():
    forecasts = _forecasts()

    with pytest.raises(ValueError, match="missing models"):
        build_forecast_error_diagnostics(forecasts, models=("model_c",))
    with pytest.raises(ValueError, match="unique model names"):
        build_forecast_error_diagnostics(
            forecasts, models=("model_a", "model_a")
        )


def test_mean_error_t_tests_match_scipy_and_holm_adjust_across_models():
    errors, _ = build_forecast_error_diagnostics(_forecasts())

    tests = build_mean_error_t_tests(errors)

    expected = errors.loc[errors["model"].eq("model_a"), "forecast_error"]
    row = tests.set_index("model").loc["model_a"]
    result = ttest_1samp(expected, popmean=0, alternative="two-sided")
    assert row["t_statistic"] == pytest.approx(result.statistic)
    assert row["p_value"] == pytest.approx(result.pvalue)
    np.testing.assert_allclose(
        tests["holm_p_value"],
        multipletests(tests["p_value"], method="holm")[1],
    )
    assert tests["n"].eq(40).all()


def test_mean_error_t_tests_reject_invalid_samples_and_alpha():
    with pytest.raises(ValueError, match="missing columns"):
        build_mean_error_t_tests(pd.DataFrame({"model": ["a"]}))
    with pytest.raises(ValueError, match="strictly between"):
        build_mean_error_t_tests(
            pd.DataFrame({"model": ["a", "a"], "forecast_error": [0.1, 0.2]}),
            alpha=1,
        )


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


def test_forecast_quantiles_are_within_model_and_summarize_error_distribution():
    errors, _ = build_forecast_error_diagnostics(_forecasts())
    classified, summary = build_forecast_quantile_diagnostics(errors)

    assert set(classified["forecast_quantile"].astype(str)) == {
        "Q1", "Q2", "Q3", "Q4", "Q5"
    }
    assert summary.groupby("model")["n"].sum().eq(40).all()
    assert summary.groupby("model")["forecast_quantile"].nunique().eq(5).all()
    assert summary.groupby("model")["n"].max().sub(
        summary.groupby("model")["n"].min()
    ).le(1).all()
    assert summary.groupby("model")["mean_forecast"].apply(
        lambda values: values.is_monotonic_increasing
    ).all()
    assert {"mean_error", "skewness", "excess_kurtosis", "MAE", "RMSE"} <= set(
        summary.columns
    )


def test_forecast_quantile_diagnostics_reject_too_few_rows_and_invalid_count():
    errors, _ = build_forecast_error_diagnostics(_forecasts())
    with pytest.raises(ValueError, match="at least two"):
        build_forecast_quantile_diagnostics(errors, quantiles=1)
    with pytest.raises(ValueError, match="fewer than"):
        build_forecast_quantile_diagnostics(errors.iloc[:4], quantiles=5)
