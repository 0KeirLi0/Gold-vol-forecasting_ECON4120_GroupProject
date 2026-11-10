import json

import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline.predictors import (
    _download_fred_series,
    _fred_api_key,
    _normalize_predictor_panel,
    download_stage2_predictors,
    load_predictor_csv,
    prepare_multivariate_dataset,
)


def make_gold_prices(periods: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    closes = 2000 * np.exp(np.arange(periods) * 0.001)
    return pd.DataFrame(
        {"date": dates, "close": closes, "volume": np.arange(periods) + 10}
    )


def test_fred_observations_are_available_next_weekday(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "observations": [
                        {"date": "2024-01-05", "value": "4.12"},
                        {"date": "2024-01-08", "value": "."},
                    ]
                }
            ).encode()

    calls = {}

    def fake_urlopen(url, timeout):
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("gold_vol_pipeline.predictors.urlopen", fake_urlopen)

    series = _download_fred_series(
        "treasury_10y_yield",
        "DGS10",
        "2024-01-01",
        "2024-01-10",
        "fake-api-key",
    )

    assert calls["timeout"] == 30
    assert "series_id=DGS10" in calls["url"]
    assert series.loc[0, "date"] == pd.Timestamp("2024-01-08")
    assert series.loc[0, "treasury_10y_yield"] == 4.12
    assert series.loc[0, "treasury_10y_yield_observation_date"] == pd.Timestamp(
        "2024-01-05"
    )


def test_fred_api_key_must_be_configured(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="FRED_API_KEY is required"):
        _fred_api_key(None)


def test_stage2_download_combines_selected_sources_and_preserves_availability(
    monkeypatch,
):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])

    def fake_yahoo(feature, ticker, start, end):
        values = [100.0, 101.0]
        return pd.DataFrame(
            {
                "date": dates,
                feature: values,
                f"{feature}_observation_date": dates,
            }
        )

    def fake_fred(feature, series_id, start, end, api_key):
        observed_date = pd.Timestamp("2024-01-02")
        available_date = pd.Timestamp("2024-01-03")
        return pd.DataFrame(
            {
                "date": [available_date],
                feature: [4.0],
                f"{feature}_observation_date": [observed_date],
            }
        )

    monkeypatch.setattr(
        "gold_vol_pipeline.predictors._download_yahoo_series", fake_yahoo
    )
    monkeypatch.setattr(
        "gold_vol_pipeline.predictors._download_fred_series", fake_fred
    )

    panel = download_stage2_predictors(
        "2024-01-01", "2024-01-05", api_key="fake-api-key"
    )

    assert {
        "dxy_close",
        "sp500_close",
        "sp500_log_return",
        "vix_close",
        "treasury_10y_yield",
        "breakeven_inflation_10y",
        "wti_spot",
    }.issubset(panel.columns)
    assert panel.attrs["fred_availability_lag"] == "one weekday after observation date"
    available_yield = panel.loc[
        panel["date"] == pd.Timestamp("2024-01-03"), "treasury_10y_yield"
    ].iloc[0]
    assert available_yield == 4.0
    assert panel["treasury_10y_yield_observation_date"].dropna().iloc[0] == pd.Timestamp(
        "2024-01-02"
    )


def test_multivariate_dataset_uses_only_predictors_available_at_origin():
    gold_prices = make_gold_prices()
    predictors = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-03")],
            "dxy_close": [110.0, 100.0],
        }
    )

    dataset = prepare_multivariate_dataset(gold_prices, predictors, horizon=2)

    earlier_origin = dataset.loc[dataset["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    same_day_origin = dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-08")
    ].iloc[0]
    assert earlier_origin["dxy_close"] == 100.0
    assert same_day_origin["dxy_close"] == 110.0
    assert earlier_origin["dxy_close_observation_date"] == pd.Timestamp("2024-01-03")
    assert "gold_volume" in dataset.columns
    assert earlier_origin["gold_volume"] == 14


def test_predictor_csv_requires_availability_dates_and_records_observation_dates(
    tmp_path,
):
    csv_path = tmp_path / "predictors.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "wti_spot": [70.5],
            "wti_spot_observation_date": ["2024-01-02"],
        }
    ).to_csv(csv_path, index=False)

    predictors = load_predictor_csv(csv_path)

    assert predictors.loc[0, "wti_spot"] == 70.5
    assert predictors.loc[0, "wti_spot_observation_date"] == pd.Timestamp(
        "2024-01-02"
    )
    assert predictors.attrs["source"] == "User-supplied CSV: predictors.csv"


def test_predictor_panel_rejects_future_source_observations():
    predictors = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "vix_close": [20.0],
            "vix_close_observation_date": ["2024-01-03"],
        }
    )

    with pytest.raises(ValueError, match="cannot be later than its availability"):
        _normalize_predictor_panel(predictors)
