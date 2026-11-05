import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline import (
    download_gold_prices,
    load_gold_prices_csv,
    prepare_dataset,
)


def make_prices(returns: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(returns) + 1)
    closes = 100 * np.exp(np.r_[0.0, np.cumsum(returns)])
    return pd.DataFrame({"date": dates, "close": closes})


def test_default_target_is_forward_five_session_realized_variance():
    returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.04, 0.01, 0.005]
    dataset = prepare_dataset(make_prices(returns))

    first = dataset.iloc[0]
    expected = sum(value**2 for value in returns[1:6])
    assert first["realized_variance"] == pytest.approx(expected)
    assert first["date"] == pd.Timestamp("2024-01-02")
    assert first["target_start_date"] == pd.Timestamp("2024-01-03")
    assert first["target_end_date"] == pd.Timestamp("2024-01-09")


def test_realized_volatility_is_square_root_of_realized_variance():
    prices = make_prices([0.01, -0.02, 0.03, 0.01])

    variance = prepare_dataset(prices, horizon=2)
    volatility = prepare_dataset(
        prices, horizon=2, target="realized_volatility"
    )

    assert volatility["realized_volatility"].to_numpy() == pytest.approx(
        np.sqrt(variance["realized_variance"].to_numpy())
    )


def test_sorts_prices_by_date_before_calculating_returns():
    prices = make_prices([0.01, 0.02, -0.01, 0.03]).iloc[::-1]

    dataset = prepare_dataset(prices, horizon=2)

    assert dataset["date"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        (
            pd.DataFrame(
                {"date": ["2024-01-01", "2024-01-01"], "close": [100, 101]}
            ),
            "at most one close",
        ),
        (
            pd.DataFrame(
                {"date": ["2024-01-01", "2024-01-02"], "close": [100, 0]}
            ),
            "must be positive",
        ),
    ],
)
def test_rejects_invalid_price_inputs(prices, message):
    with pytest.raises(ValueError, match=message):
        prepare_dataset(prices, horizon=1)


def test_rejects_nonpositive_horizon():
    with pytest.raises(ValueError, match="positive integer"):
        prepare_dataset(make_prices([0.01, 0.02]), horizon=0)


def test_loads_and_selects_csv_columns_and_records_provenance(tmp_path):
    csv_path = tmp_path / "gold.csv"
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5),
            "USD Price": [2000, 2001, 2002, 2000, 2005],
            "Notes": [""] * 5,
        }
    ).to_csv(csv_path, index=False)

    prices = load_gold_prices_csv(
        csv_path,
        date_column="Date",
        close_column="USD Price",
        source="Test provider",
    )
    dataset = prepare_dataset(prices, horizon=2)

    assert list(prices.columns) == ["date", "close"]
    assert prices.attrs["source"] == "Test provider"
    assert dataset["source"].eq("Test provider").all()
    assert len(dataset) == 2


def test_csv_loader_reports_missing_columns(tmp_path):
    csv_path = tmp_path / "gold.csv"
    pd.DataFrame({"Timestamp": ["2024-01-01"], "Value": [2000]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="CSV is missing selected column"):
        load_gold_prices_csv(csv_path)


def test_csv_loader_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Gold-price CSV not found"):
        load_gold_prices_csv(tmp_path / "missing.csv")


def test_yahoo_download_records_futures_instrument_metadata(monkeypatch):
    import yfinance as yf

    dates = pd.date_range("2024-01-01", periods=3)
    sample = pd.DataFrame(
        {"Close": [2000.0, 2002.0, 1999.0]},
        index=dates,
    )
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: sample)

    prices = download_gold_prices("2024-01-01", "2024-01-04")
    dataset = prepare_dataset(prices, horizon=1)

    assert prices.attrs["instrument"] == (
        "COMEX Gold Futures (continuous series, GC=F)"
    )
    assert dataset["instrument"].eq(
        "COMEX Gold Futures (continuous series, GC=F)"
    ).all()


def test_yahoo_empty_response_reports_futures_download_failure(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(RuntimeError, match="no gold futures prices"):
        download_gold_prices("2020-01-01", "2020-02-01")
