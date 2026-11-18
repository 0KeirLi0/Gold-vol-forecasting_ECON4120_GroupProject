import pandas as pd
import pytest

from gold_vol_pipeline.yang_zhang_calendar import summarize_yang_zhang_calendar


def _targets():
    return pd.DataFrame(
        {
            "date": [
                "2020-01-02",
                "2020-01-03",
                "2020-02-03",
                "2021-01-04",
                "2021-02-01",
                "2022-01-03",
            ],
            "daily_volatility": [0.01, 0.03, 0.02, 0.04, None, 0.05],
            "split": ["train", "train", "train", "test", "unavailable", "test"],
        }
    )


def test_calendar_summary_reports_month_of_year_and_calendar_month_stats():
    month_summary, calendar_summary = summarize_yang_zhang_calendar(_targets())

    january = month_summary.loc[
        month_summary["sample"].eq("all") & month_summary["month"].eq(1)
    ].iloc[0]
    assert january["n_targets"] == 4
    assert january["n_years"] == 3
    assert january["mean_daily_volatility"] == pytest.approx(0.0325)
    assert january["median_daily_volatility"] == pytest.approx(0.035)

    train_january = month_summary.loc[
        month_summary["sample"].eq("train") & month_summary["month"].eq(1)
    ].iloc[0]
    test_january = month_summary.loc[
        month_summary["sample"].eq("test") & month_summary["month"].eq(1)
    ].iloc[0]
    assert train_january["n_targets"] == 2
    assert test_january["n_targets"] == 2

    year_month = calendar_summary.loc[
        calendar_summary["sample"].eq("all")
        & calendar_summary["year"].eq(2020)
        & calendar_summary["month"].eq(1)
    ].iloc[0]
    assert year_month["n_targets"] == 2
    assert year_month["mean_daily_volatility"] == pytest.approx(0.02)


def test_calendar_summary_excludes_missing_targets_and_rejects_bad_values():
    targets = _targets()
    month_summary, _ = summarize_yang_zhang_calendar(targets)
    assert month_summary.loc[month_summary["sample"].eq("all"), "n_targets"].sum() == 5

    targets.loc[0, "daily_volatility"] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_yang_zhang_calendar(targets)


def test_calendar_summary_rejects_duplicate_dates_and_invalid_split():
    targets = _targets()
    targets.loc[1, "date"] = targets.loc[0, "date"]
    with pytest.raises(ValueError, match="at most one row"):
        summarize_yang_zhang_calendar(targets)

    targets = _targets()
    targets.loc[0, "split"] = "unavailable"
    with pytest.raises(ValueError, match="split='train' or split='test'"):
        summarize_yang_zhang_calendar(targets)


def test_calendar_summary_requires_observed_targets():
    targets = _targets()
    targets["daily_volatility"] = None
    with pytest.raises(ValueError, match="no valid"):
        summarize_yang_zhang_calendar(targets)
