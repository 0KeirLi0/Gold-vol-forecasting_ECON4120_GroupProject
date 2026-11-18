import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from gold_vol_pipeline.forecast_error_diagnostics import diagnose_forecast_errors


def _forecasts() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=300)
    base = np.linspace(0.01, 0.02, len(dates))
    return pd.concat(
        [
            pd.DataFrame({
                "target_date": dates,
                "model": name,
                "actual": base + 0.001 * rng.normal(size=len(dates)),
                "forecast": base - offset,
                "converged": True,
            })
            for name, offset in (("a", 0.001), ("b", 0.0))
        ],
        ignore_index=True,
    )


def test_hac_mean_variance_slope_and_holm_match_statsmodels():
    frame = _forecasts()
    # Each candidate must forecast the same realized target on a given date.
    actual = frame.loc[frame["model"].eq("a"), "actual"].to_numpy()
    frame.loc[frame["model"].eq("b"), "actual"] = actual
    result, metadata = diagnose_forecast_errors(frame, hac_lags=10)
    sample = frame.loc[frame["model"].eq("a")]
    errors = sample["actual"].to_numpy() - sample["forecast"].to_numpy()
    forecast = sample["forecast"].to_numpy()
    mean = sm.OLS(errors, np.ones((len(errors), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10}, use_t=True
    )
    x = (forecast - forecast.mean()) / forecast.std()
    slope = sm.OLS((errors - errors.mean()) ** 2, np.column_stack((np.ones(len(x)), x))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10}, use_t=True
    )
    a = result.set_index("model").loc["a"]
    assert a["mean_actual_minus_forecast"] == pytest.approx(mean.params[0])
    assert a["mean_p_value"] == pytest.approx(mean.pvalues[0])
    assert a["squared_error_slope_p_value"] == pytest.approx(slope.pvalues[1])
    np.testing.assert_allclose(
        result["mean_holm_p_value"],
        multipletests(result["mean_p_value"], method="holm")[1],
    )
    assert metadata["common_observations"] == 300


def test_common_dates_exclude_failures_and_reject_mismatched_actuals():
    frame = _forecasts()
    actual = frame.loc[frame["model"].eq("a"), "actual"].to_numpy()
    frame.loc[frame["model"].eq("b"), "actual"] = actual
    frame.loc[frame.index[300], "converged"] = False
    result, metadata = diagnose_forecast_errors(frame)
    assert result["n_common"].eq(299).all()
    assert metadata["excluded_noncommon_dates"] == 1
    frame.loc[frame.index[301], "actual"] += 0.1
    with pytest.raises(ValueError, match="Actual targets disagree"):
        diagnose_forecast_errors(frame)


def test_duplicate_and_constant_forecasts_are_rejected():
    frame = _forecasts().loc[lambda data: data["model"].eq("a")]
    with pytest.raises(ValueError, match="one row per"):
        diagnose_forecast_errors(pd.concat([frame, frame.iloc[[0]]]))
    frame = frame.assign(forecast=0.01)
    with pytest.raises(ValueError, match="constant"):
        diagnose_forecast_errors(frame)


def test_detects_positive_mean_and_forecast_scaled_error_variance():
    rng = np.random.default_rng(2026)
    forecast = rng.uniform(0.01, 0.03, 2_000)
    errors = 0.001 + rng.normal(size=len(forecast)) * (0.04 * forecast)
    frame = pd.DataFrame({
        "target_date": pd.bdate_range("2010-01-01", periods=len(forecast)),
        "model": "scaled",
        "actual": forecast + errors,
        "forecast": forecast,
        "converged": True,
    })
    result, _ = diagnose_forecast_errors(frame, hac_lags=10)
    row = result.iloc[0]
    assert row["mean_actual_minus_forecast"] > 0
    assert row["mean_reject_0.05"]
    assert row["squared_error_slope_per_forecast_sd"] > 0
    assert row["squared_error_slope_reject_0.05"]
