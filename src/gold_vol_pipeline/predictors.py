"""Acquire and align candidate daily predictors for multivariate forecasts."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .pipeline import (
    DEFAULT_HORIZON,
    DEFAULT_TARGET,
    TargetMeasure,
    download_futures_prices,
)
from .pipeline import load_gold_prices_csv, prepare_dataset
from .lseg import load_lseg_workspace_data

YAHOO_SERIES = {
    "dxy_close": ("DX-Y.NYB", "ICE U.S. Dollar Index"),
    "sp500_close": ("^GSPC", "S&P 500 price index"),
    "vix_close": ("^VIX", "CBOE Volatility Index"),
}
FRED_SERIES = {
    "treasury_10y_yield": ("DGS10", "10-year Treasury constant-maturity yield"),
    "breakeven_inflation_10y": ("T10YIE", "10-year breakeven inflation rate"),
    "wti_spot": ("DCOILWTICO", "WTI Cushing spot price"),
}
LSEG_OPTION_COLUMNS = (
    "gold_iv_1m",
    "gold_iv_2m",
    "gold_iv_3m",
    "gold_iv_6m",
    "gold_iv_1y",
    "gold_1m_risk_reversal",
    "gold_1m_butterfly",
)
PREDICTOR_COLUMNS = tuple(
    (
        *YAHOO_SERIES.keys(),
        "sp500_log_return",
        *FRED_SERIES.keys(),
        *LSEG_OPTION_COLUMNS,
    )
)


def _date_range(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if start_date >= end_date:
        raise ValueError("start must be earlier than the exclusive end date.")
    return start_date, end_date


def _normalized_dates(values: pd.Index | pd.Series) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _download_yahoo_series(
    feature: str, ticker: str, start: str, end: str
) -> pd.DataFrame:
    import yfinance as yf

    downloaded = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if downloaded.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no data for {ticker} ({feature}) "
            f"from {start} through {end}."
        )
    if "Close" not in downloaded.columns:
        raise RuntimeError(f"Yahoo Finance response for {ticker} has no Close column.")

    close = downloaded["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise RuntimeError(
                f"Yahoo Finance returned an unexpected {ticker} Close shape."
            )
        close = close.iloc[:, 0]

    dates = _normalized_dates(downloaded.index)
    values = pd.to_numeric(close.to_numpy(), errors="raise")
    valid_values = values[~pd.isna(values)]
    if not np.isfinite(valid_values).all() or (valid_values <= 0).any():
        raise RuntimeError(f"Yahoo Finance returned invalid closes for {ticker}.")
    series = pd.DataFrame(
        {
            "date": dates,
            feature: values,
            f"{feature}_observation_date": dates,
        }
    ).dropna(subset=[feature])
    if series["date"].duplicated().any():
        raise RuntimeError(f"Yahoo Finance returned duplicate dates for {ticker}.")
    if series.empty:
        raise RuntimeError(f"Yahoo Finance returned no valid closes for {ticker}.")
    return series


def _download_fred_series(
    feature: str,
    series_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_date, end_date = _date_range(start, end)
    params = urlencode(
        {
            "id": series_id,
            "cosd": start_date.date().isoformat(),
            "coed": (end_date.date() - timedelta(days=1)).isoformat(),
        }
    )
    request_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}"
    try:
        with urlopen(request_url, timeout=30) as response:
            observations = pd.read_csv(BytesIO(response.read()))
    except HTTPError as error:
        raise RuntimeError(
            f"FRED request for {series_id} failed with HTTP {error.code}."
        ) from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError(
            f"FRED request for {series_id} could not be completed."
        ) from None
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise RuntimeError(f"FRED returned an invalid CSV for {series_id}.") from None

    if observations.empty or not {"observation_date", series_id}.issubset(
        observations.columns
    ):
        raise RuntimeError(f"FRED returned no observations for {series_id}.")

    observation_dates = _normalized_dates(observations["observation_date"])
    values = pd.to_numeric(
        observations[series_id].replace(".", np.nan), errors="coerce"
    )
    numeric_values = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise RuntimeError(f"FRED returned invalid numeric values for {series_id}.")
    series = pd.DataFrame(
        {
            "date": observation_dates + pd.offsets.BDay(1),
            feature: values.to_numpy(),
            f"{feature}_observation_date": observation_dates,
        }
    ).dropna(subset=[feature])
    if series["date"].duplicated().any():
        raise RuntimeError(f"FRED returned duplicate dates for {series_id}.")
    if series.empty:
        raise RuntimeError(f"FRED returned no numeric observations for {series_id}.")
    return series


def _combine_predictor_series(series_list: list[pd.DataFrame]) -> pd.DataFrame:
    if not series_list:
        raise ValueError("At least one predictor series is required.")
    panel = series_list[0]
    for series in series_list[1:]:
        panel = panel.merge(
            series, on="date", how="outer", validate="one_to_one"
        )
    panel = panel.sort_values("date").reset_index(drop=True)
    value_columns = [column for column in PREDICTOR_COLUMNS if column in panel]
    observation_columns = [
        f"{column}_observation_date" for column in value_columns
    ]
    panel[value_columns + observation_columns] = panel[
        value_columns + observation_columns
    ].ffill()
    return panel


def download_stage2_predictors(
    start: str, end: str
) -> pd.DataFrame:
    """Download selected daily predictors for [start, end).

    Yahoo daily closes are treated as available on their observation date because
    the project forecast is issued after that date's market data are published.
    FRED observations are conservatively treated as available on the next weekday.
    This is an availability-date proxy, not a reconstruction of actual release
    timestamps or historical FRED vintages.
    The returned ``date`` is the availability date; each feature also retains its
    source observation date for audit and freshness checks.
    """
    _date_range(start, end)

    series = [
        _download_yahoo_series(feature, ticker, start, end)
        for feature, (ticker, _) in YAHOO_SERIES.items()
    ]
    sp500 = series[1]
    sp500["sp500_log_return"] = np.log(sp500["sp500_close"]).diff()
    sp500["sp500_log_return_observation_date"] = sp500[
        "sp500_close_observation_date"
    ]
    series.extend(
        _download_fred_series(feature, series_id, start, end)
        for feature, (series_id, _) in FRED_SERIES.items()
    )
    predictors = _combine_predictor_series(series)
    predictors.attrs["sources"] = {
        **{
            feature: f"Yahoo Finance {ticker} ({description})"
            for feature, (ticker, description) in YAHOO_SERIES.items()
        },
        "sp500_log_return": (
            "Close-to-close log difference calculated from Yahoo Finance ^GSPC"
        ),
        **{
            feature: f"FRED public CSV download {series_id} ({description})"
            for feature, (series_id, description) in FRED_SERIES.items()
        },
    }
    predictors.attrs["fred_availability_lag"] = "one weekday after observation date"
    return predictors


def _normalize_predictor_panel(predictors: pd.DataFrame) -> pd.DataFrame:
    if "date" not in predictors.columns:
        raise ValueError(
            "predictors must include an availability date column named 'date'."
        )
    unknown_columns = set(predictors.columns) - {
        "date",
        *PREDICTOR_COLUMNS,
        *(f"{column}_observation_date" for column in PREDICTOR_COLUMNS),
    }
    if unknown_columns:
        raise ValueError(
            "predictors contain unsupported columns: "
            + ", ".join(sorted(unknown_columns))
        )
    value_columns = [column for column in PREDICTOR_COLUMNS if column in predictors]
    if not value_columns:
        raise ValueError(
            "predictors must include at least one selected predictor column."
        )

    panel = predictors.copy()
    if panel.empty:
        raise ValueError("predictors must include at least one available observation.")
    if not panel[value_columns].notna().any().any():
        raise ValueError("predictors contain no non-missing values.")
    dates = _normalized_dates(panel["date"])
    if dates.isna().any():
        raise ValueError("predictor availability dates must not be missing.")
    panel["date"] = dates
    if panel["date"].duplicated().any():
        raise ValueError("predictors must have at most one row per availability date.")
    panel = panel.sort_values("date").reset_index(drop=True)

    for column in value_columns:
        panel[column] = pd.to_numeric(panel[column], errors="raise")
        observed_values = panel[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed_values).all():
            raise ValueError(f"{column} values must be finite or missing.")

        observation_column = f"{column}_observation_date"
        if observation_column not in panel:
            panel[observation_column] = panel["date"].where(panel[column].notna())
        else:
            observation_dates = pd.to_datetime(
                panel[observation_column], errors="raise"
            )
            if observation_dates.dt.tz is not None:
                observation_dates = observation_dates.dt.tz_localize(None)
            observation_dates = observation_dates.dt.normalize()
            missing_observation_dates = (
                panel[column].notna() & observation_dates.isna()
            )
            if missing_observation_dates.any():
                raise ValueError(
                    f"{observation_column} is required when {column} is present."
                )
            if (observation_dates > panel["date"]).fillna(False).any():
                raise ValueError(
                    f"{observation_column} cannot be later than its availability date."
                )
            panel[observation_column] = observation_dates

    observation_columns = [
        f"{column}_observation_date" for column in value_columns
    ]
    panel[value_columns + observation_columns] = panel[
        value_columns + observation_columns
    ].ffill()
    return panel.reset_index(drop=True)


def load_predictor_csv(path: str | Path) -> pd.DataFrame:
    """Load a normalized predictor CSV whose ``date`` is its availability date."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictor CSV not found: {csv_path}")
    try:
        raw = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read predictor CSV {csv_path}: {error}") from error
    predictors = _normalize_predictor_panel(raw)
    predictors.attrs["source"] = f"User-supplied CSV: {csv_path.name}"
    return predictors


def prepare_multivariate_dataset(
    gold_prices: pd.DataFrame,
    predictors: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    target: TargetMeasure = DEFAULT_TARGET,
) -> pd.DataFrame:
    """Attach the latest predictors available at each gold forecast origin.

    Predictor values are matched by availability date, never by a later
    observation date. Missing predictors remain missing; the model-development
    stage is responsible for a documented missing-data policy.
    """
    gold = prepare_dataset(gold_prices, horizon=horizon, target=target)
    panel = _normalize_predictor_panel(predictors)
    gold["date"] = pd.to_datetime(gold["date"]).astype("datetime64[ns]")
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    dataset = pd.merge_asof(
        gold.sort_values("date"),
        panel,
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    if "volume" in dataset.columns:
        dataset = dataset.rename(columns={"volume": "gold_volume"})
    return dataset.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download or import data and prepare a Stage 2 multivariate dataset."
        )
    )
    parser.add_argument(
        "--start", default="2015-01-01", help="First download date."
    )
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="Exclusive end date (default: today).",
    )
    parser.add_argument(
        "--gold-input",
        type=Path,
        help="Optional gold CSV with date, close, and optional volume columns.",
    )
    parser.add_argument(
        "--gold-source",
        default="User-supplied CSV",
        help="Provenance label for --gold-input.",
    )
    parser.add_argument(
        "--gold-instrument",
        default="Gold futures (specify contract or series)",
        help="Instrument label for --gold-input.",
    )
    parser.add_argument(
        "--predictor-input",
        type=Path,
        help="Optional normalized predictor CSV; its date column is availability date.",
    )
    parser.add_argument(
        "--lseg-input-dir",
        type=Path,
        help="Import the LSEG Workspace CSV exports from this directory.",
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--target",
        choices=("realized_variance", "realized_volatility"),
        default=DEFAULT_TARGET,
    )
    parser.add_argument("--gold-raw-output", type=Path)
    parser.add_argument(
        "--predictor-raw-output",
        type=Path,
        help="Output path for the normalized predictor panel.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.lseg_input_dir is not None:
        if args.gold_input is not None or args.predictor_input is not None:
            parser.error(
                "--lseg-input-dir cannot be combined with --gold-input or "
                "--predictor-input."
            )
        gold_prices, predictors = load_lseg_workspace_data(args.lseg_input_dir)
        default_gold_output = Path("data/raw/stage2_lseg_gold.csv")
        default_predictor_output = Path("data/raw/stage2_lseg_predictors.csv")
        default_dataset_output = Path(
            "data/processed/stage2_lseg_multivariate.csv"
        )
    else:
        if args.gold_input is None:
            gold_prices = download_futures_prices(args.start, args.end)
        else:
            gold_prices = load_gold_prices_csv(
                args.gold_input,
                source=args.gold_source,
                instrument=args.gold_instrument,
            )

        if args.predictor_input is None:
            predictors = download_stage2_predictors(args.start, args.end)
        else:
            predictors = load_predictor_csv(args.predictor_input)
        default_gold_output = Path("data/raw/stage2_gold.csv")
        default_predictor_output = Path("data/raw/stage2_predictors.csv")
        default_dataset_output = Path("data/processed/stage2_multivariate.csv")

    gold_raw_output = args.gold_raw_output or default_gold_output
    predictor_raw_output = args.predictor_raw_output or default_predictor_output
    dataset_output = args.output or default_dataset_output

    dataset = prepare_multivariate_dataset(
        gold_prices,
        predictors,
        horizon=args.horizon,
        target=args.target,
    )
    gold_raw_output.parent.mkdir(parents=True, exist_ok=True)
    predictor_raw_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    gold_prices.to_csv(gold_raw_output, index=False)
    predictors.to_csv(predictor_raw_output, index=False)
    dataset.to_csv(dataset_output, index=False)

    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_range": (
            {"start": args.start, "end_exclusive": args.end}
            if args.lseg_input_dir is None
            and (args.gold_input is None or args.predictor_input is None)
            else None
        ),
        "gold_input": (
            "GCc1.csv"
            if args.lseg_input_dir is not None
            else args.gold_input.name
            if args.gold_input is not None
            else None
        ),
        "predictor_input": (
            args.predictor_input.name if args.predictor_input is not None else None
        ),
        "lseg_input_dir": (
            str(args.lseg_input_dir) if args.lseg_input_dir is not None else None
        ),
        "gold_date_range": {
            "start": pd.to_datetime(gold_prices["date"]).min().date().isoformat(),
            "end": pd.to_datetime(gold_prices["date"]).max().date().isoformat(),
        },
        "predictor_availability_range": {
            "start": predictors["date"].min().date().isoformat(),
            "end": predictors["date"].max().date().isoformat(),
        },
        "gold_source": gold_prices.attrs.get("source", "User-supplied CSV"),
        "gold_instrument": gold_prices.attrs.get(
            "instrument", "Gold futures (specify contract or series)"
        ),
        "predictor_sources": predictors.attrs.get(
            "sources", {"source": predictors.attrs.get("source", "User-supplied CSV")}
        ),
        "fred_availability_lag": predictors.attrs.get("fred_availability_lag"),
        "predictor_availability_lag": predictors.attrs.get(
            "lseg_availability_lag",
            predictors.attrs.get(
                "fred_availability_lag",
                "as encoded by the input availability dates",
            ),
        ),
        "lseg_import": predictors.attrs.get("lseg_import"),
        "alignment": (
            "latest predictor availability date on or before each gold forecast "
            "origin; source observation dates are retained in the CSV"
        ),
        "horizon": args.horizon,
        "target": args.target,
        "files": {
            "gold_raw": str(gold_raw_output),
            "predictors_raw": str(predictor_raw_output),
            "prepared": str(dataset_output),
        },
    }
    manifest_path = dataset_output.with_name(
        f"{dataset_output.stem}_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(dataset)} forecast rows to {dataset_output}; "
        f"run metadata saved to {manifest_path}."
    )


if __name__ == "__main__":
    main()
