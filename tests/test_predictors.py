import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline.predictors import (
    _download_fred_series,
    _normalize_predictor_panel,
    download_stage2_predictors,
    load_lseg_workspace_data,
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
            return (
                "observation_date,DGS10\n"
                "2024-01-05,4.12\n"
                "2024-01-08,.\n"
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
    )

    assert calls["timeout"] == 30
    assert calls["url"].startswith(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    )
    assert "id=DGS10" in calls["url"]
    assert "cosd=2024-01-01" in calls["url"]
    assert "coed=2024-01-09" in calls["url"]
    assert series.loc[0, "date"] == pd.Timestamp("2024-01-08")
    assert series.loc[0, "treasury_10y_yield"] == 4.12
    assert series.loc[0, "treasury_10y_yield_observation_date"] == pd.Timestamp(
        "2024-01-05"
    )


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

    def fake_fred(feature, series_id, start, end):
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

    panel = download_stage2_predictors("2024-01-01", "2024-01-05")

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
    gold_prices["date"] = gold_prices["date"].astype("datetime64[s]")
    predictors = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-03")],
            "dxy_close": [110.0, 100.0],
        }
    )
    predictors["date"] = predictors["date"].astype("datetime64[us]")

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


def test_lseg_exports_load_with_observation_lag_and_optional_files(tmp_path):
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "TRDPRC_1": [2000.0, None, 2004.0, 2006.0],
        }
    ).to_csv(tmp_path / "GCc1.csv", index=False)
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "TRDPRC_1": [100.0, 101.0],
        }
    ).to_csv(tmp_path / "DXY.csv", index=False)
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "TRDPRC_1": [100.0, 101.0, 100.5],
        }
    ).to_csv(tmp_path / "SPX.csv", index=False)
    pd.DataFrame(
        {"Date": ["2024-01-02"], "TRDPRC_1": [12.0]}
    ).to_csv(tmp_path / "VIX.csv", index=False)
    pd.DataFrame(
        {"Date": ["2024-01-02"], "BID": [4.0], "ASK": [6.0]}
    ).to_csv(tmp_path / "US10YT_RR.csv", index=False)
    pd.DataFrame(
        {"Date": ["2024-01-02"], "MID_PRICE": [15.0]}
    ).to_csv(tmp_path / "XAU1MO_R.csv", index=False)
    (tmp_path / "download_manifest.json").write_text(
        '{"series":{"USBEI10Y=RR":{"status":"error"},'
        '"USCRWT=ECI":{"status":"error"}}}',
        encoding="utf-8",
    )

    gold, predictors = load_lseg_workspace_data(tmp_path)

    assert gold["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert "volume" not in gold
    assert predictors.loc[0, "date"] == pd.Timestamp("2024-01-03")
    assert predictors.loc[0, "treasury_10y_yield"] == 5.0
    assert predictors.loc[0, "gold_iv_1m"] == 15.0
    assert predictors.loc[1, "sp500_log_return"] == pytest.approx(
        np.log(101.0 / 100.0)
    )
    assert predictors.loc[1, "sp500_log_return_observation_date"] == pd.Timestamp(
        "2024-01-03"
    )
    assert predictors.attrs["lseg_availability_lag"].startswith("one weekday")
    assert predictors.attrs["lseg_import"]["gold_rows_dropped_missing_close"] == 1
    assert predictors.attrs["lseg_import"]["source_manifest_statuses"] == {
        "USBEI10Y=RR": "error",
        "USCRWT=ECI": "error",
    }
