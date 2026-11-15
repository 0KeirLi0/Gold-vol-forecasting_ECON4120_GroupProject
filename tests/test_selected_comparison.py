import pandas as pd
import pytest

from scripts.compare_selected_predictors import compare_forecasts
from gold_vol_pipeline.volatility_forecasting import EXTERNAL_MODEL_NAMES


def _forecasts():
    return pd.DataFrame([
        {
            "model": model, "origin_date": f"2024-01-0{day}",
            "target_date": f"2024-01-0{day + 1}", "actual": 2.0,
            "forecast": 3.0, "converged": True,
        }
        for model in EXTERNAL_MODEL_NAMES for day in (1, 2)
    ])


def test_compares_successful_matched_forecasts():
    baseline = _forecasts()
    selected = baseline.copy()
    selected["forecast"] = 2.5
    selected.loc[0, "converged"] = False
    result = compare_forecasts(baseline, selected)
    assert result.iloc[0]["paired_n"] == 1
    assert result.iloc[0]["selected_failure_count"] == 1
    assert result.iloc[0]["baseline_MAE"] == pytest.approx(1.0)
    assert result.iloc[0]["selected_RMSE"] == pytest.approx(0.5)
    assert result.iloc[0]["change_MAE"] == pytest.approx(-0.5)


def test_rejects_mismatched_target_dates_or_actuals():
    baseline = _forecasts()
    selected = baseline.copy()
    selected.loc[0, "target_date"] = "2024-02-01"
    with pytest.raises(ValueError, match="target dates differ"):
        compare_forecasts(baseline, selected)
    selected = baseline.copy()
    selected.loc[0, "actual"] = 3.0
    with pytest.raises(ValueError, match="actuals differ"):
        compare_forecasts(baseline, selected)
