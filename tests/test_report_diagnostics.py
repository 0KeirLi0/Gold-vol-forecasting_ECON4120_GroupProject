import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline.report_diagnostics import (
    acf_pacf_table,
    assign_forecast_quintiles,
    dickey_fuller_stationary_alternative,
    forecast_harx_with_month_dummies,
    load_shortlist_forecasts,
    month_dummy_matrix,
    q5_influence_table,
    shortlist_display_name,
    shortlist_holdout_diagnostics,
    subperiod_metrics,
    train_fraction_for_cutoff,
)


def test_shortlist_display_names_cover_report_labels():
    assert shortlist_display_name("egarchx") == "EGARCH-X"
    assert shortlist_display_name("hist_gradient_boosting_ohlc") == "HGB (OHLC)"
    assert shortlist_display_name("garch11") == "GARCH(1,1)"


def test_adf_rejects_white_noise_and_keeps_random_walk():
    rng = np.random.default_rng(7)
    stationary = rng.normal(size=400)
    walk = np.cumsum(rng.normal(size=400))

    noise = dickey_fuller_stationary_alternative(
        stationary, name="noise", sample="synthetic"
    )
    unit_root = dickey_fuller_stationary_alternative(
        walk, name="walk", sample="synthetic"
    )

    assert noise["alternative"] == "stationary"
    assert noise["autolag"] == "AIC"
    assert noise["reject_unit_root_5pct"] is True
    assert unit_root["reject_unit_root_5pct"] is False
    assert noise["used_lag"] >= 0
    assert noise["p_value"] < 0.05
    assert unit_root["p_value"] > 0.05


def test_acf_pacf_table_includes_lag_zero_and_bands():
    rng = np.random.default_rng(3)
    table = acf_pacf_table(rng.normal(size=200), name="noise", sample="synthetic", nlags=10)

    assert table.loc[0, "acf"] == pytest.approx(1.0)
    assert table.loc[0, "pacf"] == pytest.approx(1.0)
    assert set(table["lag"]) == set(range(11))
    assert (table.loc[table["lag"].gt(0), "acf_ci_lower"] < table.loc[table["lag"].gt(0), "acf"]).all()


def test_load_shortlist_forecasts_combines_baseline_and_egarchx(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=3)
    baseline_rows = []
    for model in (
        "garch11",
        "egarch",
        "garchx_ohlc_range",
        "harx_ohlc",
        "hist_gradient_boosting_ohlc",
        "ignored",
    ):
        for position, date in enumerate(dates):
            baseline_rows.append(
                {
                    "origin_date": date - pd.offsets.BDay(1),
                    "target_date": date,
                    "model": model,
                    "forecast": 0.01 + 0.001 * position,
                    "actual": 0.012,
                    "converged": True,
                }
            )
    egarchx_rows = [
        {
            "origin_date": date - pd.offsets.BDay(1),
            "target_date": date,
            "model": "egarchx",
            "forecast": 0.011,
            "actual": 0.012,
            "converged": True,
        }
        for date in dates
    ]
    baseline = tmp_path / "baseline.csv"
    egarchx = tmp_path / "egarchx.csv"
    pd.DataFrame(baseline_rows).to_csv(baseline, index=False)
    pd.DataFrame(egarchx_rows).to_csv(egarchx, index=False)

    combined = load_shortlist_forecasts(
        baseline_path=baseline, egarchx_path=egarchx
    )
    errors, summary = shortlist_holdout_diagnostics(
        combined, ljung_box_lags=(1, 2)
    )

    assert set(combined["model"].astype(str)) == {
        "garch11",
        "egarch",
        "garchx_ohlc_range",
        "egarchx",
        "harx_ohlc",
        "hist_gradient_boosting_ohlc",
    }
    assert "ignored" not in set(combined["model"].astype(str))
    assert len(errors) == 18
    assert summary["n_common"].eq(3).all()
    assert set(summary["model"]) == set(combined["model"].astype(str))


def test_load_shortlist_forecasts_rejects_missing_egarchx(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=2)
    rows = []
    for model in (
        "garch11",
        "egarch",
        "garchx_ohlc_range",
        "harx_ohlc",
        "hist_gradient_boosting_ohlc",
    ):
        for date in dates:
            rows.append(
                {
                    "origin_date": date - pd.offsets.BDay(1),
                    "target_date": date,
                    "model": model,
                    "forecast": 0.01,
                    "actual": 0.012,
                    "converged": True,
                }
            )
    baseline = tmp_path / "baseline.csv"
    egarchx = tmp_path / "egarchx.csv"
    pd.DataFrame(rows).to_csv(baseline, index=False)
    pd.DataFrame(rows[:1]).assign(model="garch11").to_csv(egarchx, index=False)
    with pytest.raises(ValueError, match="missing models"):
        load_shortlist_forecasts(baseline_path=baseline, egarchx_path=egarchx)


def test_assign_forecast_quintiles_are_equal_count():
    forecasts = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05] * 5)
    labels = assign_forecast_quintiles(forecasts)
    assert set(labels) == {1, 2, 3, 4, 5}
    assert labels.value_counts().sort_index().tolist() == [5, 5, 5, 5, 5]
    assert labels.loc[forecasts.idxmax()] == 5


def test_q5_influence_table_drops_largest_absolute_errors():
    dates = pd.bdate_range("2024-01-02", periods=25)
    rows = []
    for position, date in enumerate(dates):
        forecast = 0.01 + 0.001 * position
        actual = forecast + 0.0001
        if position >= 20:
            actual = forecast + 0.05 * (position - 19)
        rows.append(
            {
                "model": "garch11",
                "target_date": date,
                "forecast": forecast,
                "actual": actual,
                "forecast_error": actual - forecast,
            }
        )
    table = q5_influence_table(pd.DataFrame(rows), k_values=(1, 3))
    one = table.loc[table["k_dropped"].eq(1)].iloc[0]
    three = table.loc[table["k_dropped"].eq(3)].iloc[0]
    assert one["n_q5"] == 5
    assert one["dropped_dates"].endswith("2024-02-05")
    assert three["q5_variance_after_drop"] < three["q5_variance"]
    assert three["q5_variance_decline"] > one["q5_variance_decline"]
    assert 0 < one["dropped_sse_share_of_q5"] < three["dropped_sse_share_of_q5"] <= 1


def test_subperiod_metrics_restricts_dates():
    dates = pd.bdate_range("2024-01-02", periods=10)
    rows = []
    for position, date in enumerate(dates):
        rows.append(
            {
                "model": "harx_ohlc",
                "target_date": date,
                "forecast": 0.01,
                "actual": 0.01 + 0.002 * (position >= 5),
                "forecast_error": 0.002 * (position >= 5),
            }
        )
    table = subperiod_metrics(
        pd.DataFrame(rows),
        splits={
            "early": ("2024-01-02", "2024-01-08"),
            "late": ("2024-01-09", "2024-01-15"),
        },
    )
    early = table.loc[table["split"].eq("early")].iloc[0]
    late = table.loc[table["split"].eq("late")].iloc[0]
    assert early["n"] == 5
    assert late["n"] == 5
    assert early["MAE"] == pytest.approx(0.0)
    assert late["MAE"] == pytest.approx(0.002)


def test_train_fraction_keeps_cutoff_as_last_training_date():
    dates = pd.bdate_range("2023-01-02", periods=100)
    fraction = train_fraction_for_cutoff(dates, "2023-03-15")
    split = int(np.floor(len(dates) * fraction))
    assert dates[split - 1] == pd.Timestamp("2023-03-15")
    assert dates[split] > pd.Timestamp("2023-03-15")


def test_month_dummy_matrix_omits_january():
    dates = pd.to_datetime(["2024-01-15", "2024-02-15", "2024-12-15"])
    dummies = month_dummy_matrix(dates)
    assert dummies.shape == (3, 11)
    assert dummies[0].sum() == 0
    assert dummies[1, 0] == 1
    assert dummies[2, 10] == 1


def test_harx_month_dummy_forecast_is_positive():
    rng = np.random.default_rng(4)
    n = 40
    history = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-02", periods=n),
            "daily_volatility": 0.01 + 0.002 * np.sin(np.arange(n) / 6),
            "close_return": rng.normal(0, 0.01, n),
            "overnight_log_return": rng.normal(0, 0.004, n),
            "open_close_log_return": rng.normal(0, 0.006, n),
            "high_low_log_range": np.abs(rng.normal(0.012, 0.002, n)),
        }
    )
    january = forecast_harx_with_month_dummies(history, "2020-02-28")
    february = forecast_harx_with_month_dummies(history, "2020-03-02")
    assert january > 0
    assert february > 0
    assert np.isfinite([january, february]).all()
    assert january != february

