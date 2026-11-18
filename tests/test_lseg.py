import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline.lseg import load_lseg_workspace_data


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
