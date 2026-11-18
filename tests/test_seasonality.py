import numpy as np
import pandas as pd

from gold_vol_pipeline.seasonality import (
    build_seasonality_data,
    create_seasonality_figures,
    run_seasonality_tests,
    summarize_calendar_groups,
)


def test_build_seasonality_data_attaches_calendar_fields():
    prepared = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "daily_volatility": [0.01, np.nan, 0.02],
        }
    )
    result = build_seasonality_data(prepared)
    assert len(result) == 2
    assert result["month"].tolist() == [1, 1]
    assert result["weekday"].tolist() == ["Monday", "Wednesday"]
    assert np.isnan(result["lag1_volatility"].iloc[0])
    assert result["lag1_volatility"].iloc[1] == 0.01


def test_calendar_summaries_and_hac_tests(tmp_path):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    prepared = pd.DataFrame(
        {
            "date": dates,
            "daily_volatility": rng.uniform(0.005, 0.015, len(dates)),
        }
    )
    observations = build_seasonality_data(prepared)
    month = summarize_calendar_groups(observations, "month")
    weekday = summarize_calendar_groups(observations, "weekday")
    tests = run_seasonality_tests(observations, hac_lags=(5,))

    assert 1 <= len(month) <= 12
    assert len(weekday) == 5
    assert {"month", "weekday"} <= set(tests["category"])
    assert {"month_only", "month_and_lag1",
            "weekday_only", "weekday_and_lag1"} <= set(tests["model"])
    assert tests["n"].gt(0).all()

    figures = create_seasonality_figures(
        observations,
        month,
        weekday,
        tmp_path,
    )
    assert set(figures) == {
        "month_means",
        "weekday_means",
        "month_year_heatmap",
    }
    assert all((tmp_path / name).stat().st_size > 0 for name in [
        "month_means.png",
        "weekday_means.png",
        "month_year_heatmap.png",
    ])
