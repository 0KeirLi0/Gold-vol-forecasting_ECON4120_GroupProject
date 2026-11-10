import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline.diagnostics import check_return_assumptions


def test_diagnostics_return_expected_model_checks():
    returns = np.random.default_rng(42).normal(0, 0.01, size=250)

    report = check_return_assumptions(returns)

    assert report["diagnostic"].tolist() == [
        "sample_summary",
        "mean_zero_t_test",
        "returns_adf_stationarity",
        "squared_returns_adf_stationarity",
        "returns_ljung_box",
        "squared_returns_ljung_box",
        "arch_lm",
        "return_normality_jarque_bera",
    ]
    assert report.loc[1:, "p_value"].between(0, 1).all()


def test_diagnostics_detect_volatility_clustering_signal():
    rng = np.random.default_rng(7)
    shocks = rng.normal(size=500)
    variance = np.empty_like(shocks)
    variance[0] = 1
    for index in range(1, len(shocks)):
        variance[index] = 0.05 + 0.9 * variance[index - 1] + 0.05 * shocks[index - 1] ** 2
    returns = shocks * np.sqrt(variance) * 0.01

    report = check_return_assumptions(returns)
    arch_result = report.loc[report["diagnostic"] == "arch_lm", "result"].iloc[0]

    assert arch_result == "evidence against null"


@pytest.mark.parametrize(
    ("returns", "message"),
    [
        ([0.01] * 19, "at least 20"),
        ([0.01] * 20, "non-zero variance"),
        ([0.01] * 19 + [np.nan], "finite numeric"),
    ],
)
def test_diagnostics_reject_data_that_cannot_support_tests(returns, message):
    with pytest.raises(ValueError, match=message):
        check_return_assumptions(returns)


def test_diagnostics_reject_invalid_test_options():
    returns = pd.Series(np.random.default_rng(2).normal(size=50))

    with pytest.raises(ValueError, match="lags"):
        check_return_assumptions(returns, lags=0)
    with pytest.raises(ValueError, match="alpha"):
        check_return_assumptions(returns, alpha=1)
