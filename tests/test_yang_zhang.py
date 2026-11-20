import numpy as np
import pandas as pd
import pytest
import json

import gold_vol_pipeline.volatility_forecasting as volatility_forecasting
from gold_vol_pipeline.lseg import align_lseg_vix, align_selected_forecast_predictors
from gold_vol_pipeline.volatility_forecasting import (
    _lagged_design,
    _variance_recursion,
    EXTERNAL_MODEL_NAMES,
    SELECTED_PREDICTORS,
    evaluate_daily_volatility,
    forecast_daily_volatility,
    save_daily_volatility_evaluation,
)
from gold_vol_pipeline.segmented_forecasting import (
    SEGMENTED_MODELS,
    TRIGGERS,
    evaluate_segmented_daily_volatility,
    segmented_forecast,
)
from gold_vol_pipeline.yang_zhang import (
    add_chronological_split,
    prepare_yang_zhang_dataset,
)


def test_saved_evaluation_uses_stage2_names_and_records_input_hash(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("source data", encoding="utf-8")
    output = tmp_path / "output"
    save_daily_volatility_evaluation(
        pd.DataFrame({"forecast": [0.1]}),
        pd.DataFrame({"MAE": [0.01]}),
        {"target": "daily_volatility"},
        output,
        input_path=source,
    )
    assert {path.name for path in output.iterdir()} == {
        "stage2_yz_forecasts.csv",
        "stage2_yz_summary.csv",
        "stage2_yz_manifest.json",
    }
    manifest = json.loads((output / "stage2_yz_manifest.json").read_text())
    assert manifest["input_path"] == str(source)
    assert len(manifest["input_sha256"]) == 64


def test_experimental_forecasts_are_leakage_safe_and_report_qlike():
    prepared = prepare_yang_zhang_dataset(_ohlc_data(size=120), window=2)
    results, summary, manifest = evaluate_daily_volatility(
        prepared,
        model_names=("hist_gradient_boosting_ohlc",),
        rolling_windows=(50,),
        bias_window=3,
        hgb_transform_experiments=True,
        progress=False,
    )
    first_date = results["target_date"].min()
    first_origin = results.loc[results["target_date"].eq(first_date)]
    base_first = first_origin.loc[
        first_origin["model"].eq("hist_gradient_boosting_ohlc"), "forecast"
    ].item()
    bias_first = first_origin.loc[
        first_origin["model"].eq("hist_gradient_boosting_ohlc__bias_3"), "forecast"
    ].item()
    assert bias_first == pytest.approx(base_first)

    second_date = results["target_date"].sort_values().unique()[1]
    second_origin = results.loc[results["target_date"].eq(second_date)]
    base_second = second_origin.loc[
        second_origin["model"].eq("hist_gradient_boosting_ohlc"), "forecast"
    ].item()
    first_actual = first_origin.loc[
        first_origin["model"].eq("hist_gradient_boosting_ohlc"), "actual"
    ].item()
    bias_second = second_origin.loc[
        second_origin["model"].eq("hist_gradient_boosting_ohlc__bias_3"), "forecast"
    ].item()
    assert bias_second == pytest.approx(
        max(base_second + first_actual - base_first, 1e-12)
    )

    rolling = results.loc[
        results["model"].eq("hist_gradient_boosting_ohlc__rolling_50")
    ]
    assert rolling["history_count"].eq(50).all()
    assert {
        "hist_gradient_boosting_ohlc__smearing",
        "hist_gradient_boosting_ohlc__gamma",
    } <= set(results["model"])
    assert np.isfinite(summary["common_QLIKE"]).all()
    assert {"low_actual_tercile_MAE", "middle_actual_tercile_MAE", "high_actual_tercile_MAE"} <= set(summary)
    assert manifest["rolling_windows"] == [50]
    assert manifest["bias_calibration_window"] == 3
    regime_cutoffs = manifest["regime_cutoffs"]
    assert regime_cutoffs["low_middle"] <= regime_cutoffs["middle_high"]
    assert regime_cutoffs["units"] == "unannualized daily volatility"
    assert "rank(method='first'" in manifest["regime_cutoffs"]["assignment"]


def test_segmented_forecast_uses_separate_regime_sample_and_trigger(monkeypatch):
    history = prepare_yang_zhang_dataset(_ohlc_data(size=180)).dropna(
        subset=["daily_volatility"]
    )
    seen = []

    def fake_forecast(model, data, **kwargs):
        seen.append(len(data))
        return 0.5

    monkeypatch.setattr(
        "gold_vol_pipeline.segmented_forecasting.forecast_daily_volatility",
        fake_forecast,
    )
    forecast, metadata = segmented_forecast(
        "garch11",
        history,
        trigger="forecast_trigger",
    )

    assert forecast == pytest.approx(0.5)
    assert metadata["segment"] == "extreme"
    assert metadata["pilot_forecast"] == pytest.approx(0.5)
    assert metadata["parameter_sample_count"] == seen[-1]
    assert seen[0] == len(history)
    assert seen[-1] < len(history)


def test_segmented_evaluation_runs_both_triggers_without_target_leakage():
    prepared = prepare_yang_zhang_dataset(_ohlc_data(size=260))
    results, summary, manifest = evaluate_segmented_daily_volatility(
        prepared,
        model_names=("hist_gradient_boosting_ohlc",),
        triggers=TRIGGERS,
        external_predictors=(),
        progress=False,
    )

    assert set(results["trigger"]) == set(TRIGGERS)
    assert set(summary["trigger"]) == set(TRIGGERS)
    assert manifest["threshold_quantile"] == pytest.approx(0.8)
    assert manifest["test_forecast_method"].startswith("sequential one-step")
    assert results["target_date"].min() > results["origin_date"].min()


def _ohlc_data(size: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size)))
    open_price = close * np.exp(rng.normal(0, 0.003, size))
    high = np.maximum(open_price, close) * np.exp(rng.uniform(0.001, 0.01, size))
    low = np.minimum(open_price, close) * np.exp(-rng.uniform(0.001, 0.01, size))
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=size),
            "TRDPRC_1": close,
            "OPEN_PRC": open_price,
            "HIGH_1": high,
            "LOW_1": low,
        }
    )


def test_yang_zhang_matches_ohlc_formula_for_two_observations():
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3),
            "TRDPRC_1": [101.0, 102.0, 103.0],
            "OPEN_PRC": [100.0, 101.5, 102.5],
            "HIGH_1": [102.0, 103.0, 104.0],
            "LOW_1": [99.0, 100.0, 101.5],
        }
    )
    result = prepare_yang_zhang_dataset(raw, window=2)

    overnight = np.log(np.array([101.5 / 101.0, 102.5 / 102.0]))
    open_close = np.log(np.array([102.0 / 101.5, 103.0 / 102.5]))
    high = np.array([103.0, 104.0])
    low = np.array([100.0, 101.5])
    opening = np.array([101.5, 102.5])
    closing = np.array([102.0, 103.0])
    rogers_satchell = (
        np.log(high / opening) * np.log(high / closing)
        + np.log(low / opening) * np.log(low / closing)
    )
    k = 0.34 / (1.34 + 3 / 1)
    expected_variance = (
        np.var(overnight, ddof=1)
        + k * np.var(open_close, ddof=1)
        + (1 - k) * np.mean(rogers_satchell)
    )

    assert result.loc[2, "yang_zhang_variance"] == pytest.approx(expected_variance)
    assert result.loc[2, "daily_volatility"] == pytest.approx(np.sqrt(expected_variance))
    assert pd.isna(result.loc[1, "daily_volatility"])


def test_default_yang_zhang_window_is_two_observations():
    result = prepare_yang_zhang_dataset(_ohlc_data(size=3))

    assert result.loc[:1, "daily_volatility"].isna().all()
    assert result.loc[2, "daily_volatility"] > 0
    assert result.loc[2, "window_start_date"] == result.loc[0, "date"]


def test_yang_zhang_window_must_be_at_least_two():
    with pytest.raises(ValueError, match="at least 2 observations"):
        prepare_yang_zhang_dataset(_ohlc_data(size=3), window=1)


def test_incomplete_ohlc_is_not_imputed_and_split_is_chronological():
    raw = _ohlc_data()
    raw.loc[50, "HIGH_1"] = np.nan

    prepared = prepare_yang_zhang_dataset(raw, window=5)
    split = add_chronological_split(prepared)
    valid = split.loc[split["daily_volatility"].notna()]
    train_dates = valid.loc[valid["split"].eq("train"), "date"]
    test_dates = valid.loc[valid["split"].eq("test"), "date"]

    assert not bool(split.loc[50, "complete_ohlc"])
    assert pd.isna(split.loc[50, "daily_volatility"])
    assert (train_dates.max() < test_dates.min())
    assert len(train_dates) == int(np.floor(len(valid) * 0.8))


def test_sequential_holdout_uses_prior_test_actuals_without_target_leakage():
    prepared = prepare_yang_zhang_dataset(_ohlc_data(), window=5)
    prepared = add_chronological_split(prepared)
    results, summary, manifest = evaluate_daily_volatility(
        prepared,
        model_names=("yang_zhang_persistence", "expanding_mean"),
        progress=False,
    )
    valid = prepared.loc[prepared["daily_volatility"].notna()].reset_index(drop=True)
    split_position = int(np.floor(len(valid) * 0.8))
    persistence = results.loc[
        results["model"].eq("yang_zhang_persistence")
    ].reset_index(drop=True)

    assert manifest["training_observations"] == split_position
    assert manifest["test_observations"] == len(valid) - split_position
    assert persistence.loc[0, "forecast"] == pytest.approx(
        valid.loc[split_position - 1, "daily_volatility"]
    )
    assert persistence.loc[1, "forecast"] == pytest.approx(
        persistence.loc[0, "actual"]
    )
    assert set(summary["model"]) == {"yang_zhang_persistence", "expanding_mean"}
    assert summary["common_n"].eq(len(valid) - split_position).all()


def test_vix_availability_is_next_weekday_and_asof(tmp_path):
    path = tmp_path / "VIX.csv"
    pd.DataFrame(
        {"Date": ["2024-01-05", "2024-01-09"], "TRDPRC_1": [10.0, 20.0]}
    ).to_csv(path, index=False)
    target = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]),
            "daily_volatility": [0.1, 0.2, 0.3],
        }
    )
    result = align_lseg_vix(target, path)
    assert result["vix_close"].tolist() == [10.0, 10.0, 20.0]
    assert result["vix_close_observation_date"].tolist() == list(
        pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-09"])
    )
    assert (result["vix_available_date"] <= result["date"]).all()


def test_selected_predictor_alignment_uses_next_weekday(tmp_path):
    for filename, field in (
        ("DXY.csv", "TRDPRC_1"),
        ("XAU1MO_R.csv", "MID_PRICE"),
        ("XAU1YO_R.csv", "MID_PRICE"),
    ):
        pd.DataFrame({"Date": ["2024-01-05", "2024-01-09"], field: [10., 20.]}).to_csv(
            tmp_path / filename, index=False
        )
    targets = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]),
        "daily_volatility": [0.1, 0.2, 0.3],
    })
    aligned = align_selected_forecast_predictors(targets, tmp_path)
    for column in SELECTED_PREDICTORS:
        assert aligned[column].tolist() == [10., 10., 20.]
        assert (aligned[f"{column}_available_date"] <= aligned["date"]).all()
    assert "vix_close" not in aligned


@pytest.mark.parametrize(("model", "helper"), [
    ("harx_ohlc", "_linear_forecast"),
    ("ridge_ohlc", "_linear_forecast"),
    ("lasso_ohlc", "_linear_forecast"),
    ("hist_gradient_boosting_ohlc", "_linear_forecast"),
    ("garchx_ohlc_range", "_conditional_variance_forecast"),
    ("egarchx", "_egarchx_forecast"),
    ("var_ohlc", "_var_forecast"),
])
def test_selected_predictors_forwarded_to_all_external_models(monkeypatch, model, helper):
    history = prepare_yang_zhang_dataset(_ohlc_data()).dropna(subset=["daily_volatility"])
    for column in SELECTED_PREDICTORS:
        history[column] = 10.
    seen = []
    def fake_forecast(data, **kwargs):
        seen.append(kwargs)
        return 0.01
    monkeypatch.setattr(volatility_forecasting, helper, fake_forecast)
    forecast_daily_volatility(
        model, history, auto_arima_order=(1, 0, 0),
        external_predictors=SELECTED_PREDICTORS,
    )
    assert seen[0]["external_predictors"] == SELECTED_PREDICTORS
    if model != "egarchx":
        assert not seen[0]["include_vix"]
    assert model in EXTERNAL_MODEL_NAMES


def test_selected_features_are_from_origin_not_next_target():
    history = prepare_yang_zhang_dataset(_ohlc_data()).dropna(subset=["daily_volatility"]).reset_index(drop=True)
    for i, column in enumerate(SELECTED_PREDICTORS):
        history[column] = np.arange(len(history)) + i + 1
    features, _, latest = _lagged_design(
        history, include_ohlc=True, external_predictors=SELECTED_PREDICTORS,
    )
    np.testing.assert_array_equal(features[0, -3:], history.loc[21, list(SELECTED_PREDICTORS)].to_numpy(dtype=float))
    np.testing.assert_array_equal(latest[0, -3:], history.loc[len(history)-1, list(SELECTED_PREDICTORS)].to_numpy(dtype=float))


def test_variance_recursion_matches_sequential_calculation():
    rng = np.random.default_rng(42)
    forcing = rng.uniform(0.01, 2, 4000)
    for beta in (0., 0.7, 0.999):
        expected = np.empty(len(forcing) + 1)
        expected[0] = 0.4
        for i, value in enumerate(forcing, 1):
            expected[i] = value + beta * expected[i - 1]
        np.testing.assert_allclose(_variance_recursion(forcing, beta, 0.4), expected, rtol=1e-14)


def test_selected_predictor_on_target_date_does_not_enter_its_forecast():
    prepared = prepare_yang_zhang_dataset(_ohlc_data())
    for i, column in enumerate(SELECTED_PREDICTORS):
        prepared[column] = np.linspace(10 + i, 30 + i, len(prepared))
    changed = prepared.copy()
    valid_index = changed.index[changed["daily_volatility"].notna()]
    first_test_index = valid_index[int(np.floor(len(valid_index) * 0.8))]
    changed.loc[first_test_index, list(SELECTED_PREDICTORS)] = 1000.
    before, _, _ = evaluate_daily_volatility(
        prepared, model_names=("ridge_ohlc",), external_predictors=SELECTED_PREDICTORS, progress=False
    )
    after, _, _ = evaluate_daily_volatility(
        changed, model_names=("ridge_ohlc",), external_predictors=SELECTED_PREDICTORS, progress=False
    )
    assert before.iloc[0]["forecast"] == pytest.approx(after.iloc[0]["forecast"])
    assert before.iloc[1]["forecast"] != pytest.approx(after.iloc[1]["forecast"])


def test_vix_features_are_origin_aligned_and_do_not_change_univariate_models():
    history = prepare_yang_zhang_dataset(_ohlc_data(), window=2).dropna(
        subset=["daily_volatility"]
    ).reset_index(drop=True)
    history["vix_close"] = np.linspace(10, 30, len(history))
    features, targets, forecast_features = _lagged_design(
        history, include_ohlc=True, include_vix=True
    )
    assert features[0, -1] == pytest.approx(history.loc[21, "vix_close"])
    assert targets[0] == pytest.approx(np.log(history.loc[22, "daily_volatility"]))
    assert forecast_features[0, -1] == pytest.approx(history.loc[len(history)-1, "vix_close"])
    assert forecast_daily_volatility(
        "yang_zhang_persistence", history, auto_arima_order=(1, 0, 0), include_vix=True
    ) == pytest.approx(history.iloc[-1]["daily_volatility"])


def test_vix_evaluation_rejects_missing_values():
    prepared = prepare_yang_zhang_dataset(_ohlc_data(), window=2)
    prepared["vix_close"] = 20.0
    prepared.loc[50, "vix_close"] = np.nan
    with pytest.raises(ValueError, match="VIX must contain finite positive"):
        evaluate_daily_volatility(
            prepared, model_names=("harx_ohlc",), include_vix=True, progress=False
        )


@pytest.mark.parametrize(
    ("model", "helper"),
    [
        ("harx_ohlc", "_linear_forecast"),
        ("ridge_ohlc", "_linear_forecast"),
        ("lasso_ohlc", "_linear_forecast"),
        ("hist_gradient_boosting_ohlc", "_linear_forecast"),
        ("garchx_ohlc_range", "_conditional_variance_forecast"),
        ("var_ohlc", "_var_forecast"),
    ],
)
def test_vix_is_forwarded_to_each_external_model(monkeypatch, model, helper):
    history = prepare_yang_zhang_dataset(_ohlc_data(), window=2).dropna(
        subset=["daily_volatility"]
    )
    history["vix_close"] = 20.0
    seen = []

    def fake_forecast(data, **kwargs):
        assert "vix_close" in data
        seen.append(kwargs["include_vix"])
        return 0.01

    monkeypatch.setattr(volatility_forecasting, helper, fake_forecast)
    forecast_daily_volatility(
        model, history, auto_arima_order=(1, 0, 0), include_vix=True
    )
    assert seen == [True]


def test_vix_on_target_date_does_not_enter_its_own_forecast():
    prepared = prepare_yang_zhang_dataset(_ohlc_data(), window=2)
    prepared["vix_close"] = np.linspace(10, 30, len(prepared))
    changed = prepared.copy()
    valid_index = changed.index[changed["daily_volatility"].notna()]
    first_test_index = valid_index[int(np.floor(len(valid_index) * 0.8))]
    changed.loc[first_test_index, "vix_close"] = 1_000.0
    before, _, _ = evaluate_daily_volatility(
        prepared, model_names=("harx_ohlc",), include_vix=True, progress=False
    )
    after, _, _ = evaluate_daily_volatility(
        changed, model_names=("harx_ohlc",), include_vix=True, progress=False
    )
    assert before.iloc[0]["forecast"] == pytest.approx(after.iloc[0]["forecast"])
    assert before.iloc[1]["forecast"] != pytest.approx(after.iloc[1]["forecast"])


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (pd.DataFrame({"Date": ["2024-01-01"]}), "missing required columns"),
        (
            pd.DataFrame(
                {
                    "Date": ["2024-01-01"],
                    "TRDPRC_1": [100.0],
                    "OPEN_PRC": [101.0],
                    "HIGH_1": [100.5],
                    "LOW_1": [99.0],
                }
            ),
            "satisfy low <= open/close <= high",
        ),
    ],
)
def test_rejects_invalid_ohlc_data(raw, message):
    with pytest.raises(ValueError, match=message):
        prepare_yang_zhang_dataset(raw, window=2)
