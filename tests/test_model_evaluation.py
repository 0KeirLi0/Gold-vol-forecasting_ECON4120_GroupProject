import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from gold_vol_pipeline.model_evaluation import (
    adf_unit_root_test,
    compare_candidate_models,
    ljung_box_residual_tests,
    plot_acf_pacf,
    plot_residual_panel,
    plot_series_transformations,
    save_residuals,
    transform_price_series,
)


def test_price_transformations_include_levels_differences_and_returns():
    prices = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.date_range("2024-01-01", periods=3),
    )

    transformed = transform_price_series(prices)

    assert transformed["price"].tolist() == prices.tolist()
    assert transformed["log_price"].iloc[-1] == pytest.approx(np.log(121.0))
    assert transformed["first_difference"].iloc[1] == pytest.approx(10.0)
    assert transformed["log_difference"].iloc[1] == pytest.approx(np.log(1.1))
    assert transformed["percent_change"].iloc[1] == pytest.approx(0.1)
    assert transformed["squared_log_return"].iloc[1] == pytest.approx(np.log(1.1) ** 2)


def test_price_transformations_reject_nonpositive_or_missing_values():
    with pytest.raises(ValueError, match="strictly positive"):
        transform_price_series([1.0, 0.0])
    with pytest.raises(ValueError, match="finite numeric"):
        transform_price_series([1.0, np.nan])


def test_adf_reports_unit_root_test_and_matching_critical_value():
    returns = np.random.default_rng(42).normal(size=100)

    result = adf_unit_root_test(returns, name="returns")
    expected_critical_value = adfuller(returns, autolag="AIC")[4]["5%"]

    assert result.loc[0, "series"] == "returns"
    assert 0 <= result.loc[0, "p_value"] <= 1
    assert result.loc[0, "nobs"] > 0
    assert "critical_value_5%" in result
    assert result.loc[0, "critical_value_5%"] == pytest.approx(
        expected_critical_value
    )
    assert isinstance(result.loc[0, "reject_unit_root"], (bool, np.bool_))


def test_candidate_comparison_returns_information_criteria_and_residuals():
    rng = np.random.default_rng(7)
    returns = pd.Series(
        rng.normal(0, 0.01, size=160),
        index=pd.date_range("2020-01-01", periods=160),
        name="log_return",
    )

    evaluation = compare_candidate_models(
        returns,
        ar_orders=(1,),
        ma_orders=(1,),
        arma_orders=((1, 1),),
        arch_orders=(1,),
        garch_orders=((1, 1),),
        egarch_orders=((1, 1),),
    )

    comparison = evaluation.model_comparison
    assert set(comparison["model_family"]) == {
        "AR",
        "MA",
        "ARMA",
        "ARCH",
        "GARCH",
        "EGARCH",
    }
    assert comparison["AIC"].notna().all()
    assert comparison["BIC"].notna().all()
    assert comparison["model_name"].is_unique
    assert evaluation.residuals.shape == (len(returns), 6)
    assert evaluation.standardized_residuals.shape == (len(returns), 6)


def test_ljung_box_tests_cover_residual_and_squared_standardized_series():
    returns = np.random.default_rng(12).normal(0, 0.01, size=100)
    evaluation = compare_candidate_models(
        returns,
        ar_orders=(1,),
        ma_orders=(),
        arma_orders=(),
        arch_orders=(),
        garch_orders=((1, 1),),
        egarch_orders=(),
    )

    tests = ljung_box_residual_tests(evaluation, lags=(5, 10))

    assert len(tests) == 8
    assert set(tests["series_tested"]) == {
        "residuals",
        "squared_standardized_residuals",
    }
    assert tests["model_df"].eq(0).all()
    assert tests["p_value"].between(0, 1).all()


def test_residuals_can_be_saved_and_plotted(tmp_path):
    pytest.importorskip("matplotlib")
    returns = np.random.default_rng(19).normal(0, 0.01, size=100)
    evaluation = compare_candidate_models(
        returns,
        ar_orders=(1,),
        ma_orders=(),
        arma_orders=(),
        arch_orders=(),
        garch_orders=(),
        egarch_orders=(),
    )
    residual_path = tmp_path / "residuals.csv"
    plot_path = tmp_path / "residuals.png"

    save_residuals(evaluation, residual_path)
    figure, _ = plot_residual_panel(evaluation, output_path=plot_path)
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(figure)
    transformed = transform_price_series(
        pd.Series(np.exp(np.cumsum(returns) + 5), index=pd.RangeIndex(len(returns)))
    )
    figure, _ = plot_series_transformations(transformed)
    plt.close(figure)
    figure, _ = plot_acf_pacf(returns, lags=20)
    plt.close(figure)

    saved = pd.read_csv(residual_path, index_col=0)
    assert saved.shape == evaluation.residuals.shape
    assert plot_path.exists()
