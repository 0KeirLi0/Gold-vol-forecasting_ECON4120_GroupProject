"""Load LSEG Workspace CSV exports into the project's gold and predictor schemas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_PRICE_SERIES = {
    "DXY.csv": ("dxy_close", "TRDPRC_1", "USD Dollar Index"),
    "SPX.csv": ("sp500_close", "TRDPRC_1", "S&P 500 price index"),
    "VIX.csv": ("vix_close", "TRDPRC_1", "CBOE Volatility Index"),
}
_OPTION_SERIES = {
    "XAU1MO_R.csv": ("gold_iv_1m", "MID_PRICE", "Gold 1M implied volatility"),
    "XAU2MO_R.csv": ("gold_iv_2m", "MID_PRICE", "Gold 2M implied volatility"),
    "XAU3MO_R.csv": ("gold_iv_3m", "MID_PRICE", "Gold 3M implied volatility"),
    "XAU6MO_R.csv": ("gold_iv_6m", "MID_PRICE", "Gold 6M implied volatility"),
    "XAU1YO_R.csv": ("gold_iv_1y", "MID_PRICE", "Gold 1Y implied volatility"),
    "XAU1MRR.csv": (
        "gold_1m_risk_reversal",
        "MID_PRICE",
        "Gold 1M risk reversal",
    ),
    "XAU1MBF.csv": ("gold_1m_butterfly", "MID_PRICE", "Gold 1M butterfly"),
}
_OPTIONAL_SERIES = {
    "US10YT_RR.csv": (
        "treasury_10y_yield",
        ("BID", "ASK"),
        "10-year US Treasury yield",
    ),
    "USBEI10Y_RR.csv": (
        "breakeven_inflation_10y",
        ("BID", "ASK"),
        "10-year breakeven inflation quote",
    ),
    "USCRWT_ECI.csv": ("wti_spot", "TRDPRC_1", "WTI spot price"),
}


def _read_export(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"LSEG export CSV not found: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read LSEG export {path}: {error}") from error
    if "Date" not in frame.columns:
        raise ValueError(f"LSEG export {path.name} must include a 'Date' column.")

    dates = pd.DatetimeIndex(pd.to_datetime(frame["Date"], errors="raise"))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    if dates.isna().any():
        raise ValueError(f"LSEG export {path.name} has missing dates.")
    if dates.duplicated().any():
        raise ValueError(f"LSEG export {path.name} has duplicate dates.")
    frame = frame.drop(columns="Date").copy()
    frame.insert(0, "Date", dates)
    return frame.sort_values("Date").reset_index(drop=True)


def _numeric_field(frame: pd.DataFrame, field: str, path: Path) -> pd.Series:
    if field not in frame.columns:
        raise ValueError(
            f"LSEG export {path.name} is missing field {field!r}; "
            f"available fields: {', '.join(map(str, frame.columns))}."
        )
    try:
        values = pd.to_numeric(frame[field], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"LSEG field {field!r} in {path.name} must be numeric."
        ) from error
    finite_values = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_values).all():
        raise ValueError(f"LSEG field {field!r} in {path.name} must be finite.")
    return values


def _make_predictor_series(
    input_dir: Path,
    filename: str,
    feature: str,
    fields: str | tuple[str, str],
    description: str,
) -> tuple[pd.DataFrame | None, str | None]:
    path = input_dir / filename
    if not path.is_file():
        return None, None

    frame = _read_export(path)
    if isinstance(fields, tuple):
        bid = _numeric_field(frame, fields[0], path)
        ask = _numeric_field(frame, fields[1], path)
        values = (bid + ask) / 2
        source_fields = f"{fields[0]}/{fields[1]} midpoint"
    else:
        values = _numeric_field(frame, fields, path)
        source_fields = fields

    dates = pd.DatetimeIndex(frame["Date"])
    series = pd.DataFrame(
        {
            "date": dates + pd.offsets.BDay(1),
            feature: values.to_numpy(),
            f"{feature}_observation_date": dates,
        }
    ).dropna(subset=[feature])
    if series.empty:
        raise ValueError(f"LSEG export {path.name} has no numeric {source_fields} values.")
    if feature in {"dxy_close", "sp500_close", "vix_close"}:
        if (series[feature] <= 0).any():
            raise ValueError(f"LSEG prices in {path.name} must be positive.")
    series.attrs["source"] = f"LSEG Workspace {path.name} ({description}; {source_fields})"
    return series, series.attrs["source"]


def _read_source_manifest(input_dir: Path) -> dict[str, object]:
    path = input_dir / "download_manifest.json"
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read LSEG manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"LSEG manifest {path} must contain a JSON object.")
    return manifest


def load_lseg_workspace_data(
    input_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load available LSEG exports, keeping only GCc1 for the gold target.

    Predictor observations are conservatively considered available on the next
    weekday after their source date. Source observation dates remain in the
    returned panel. Gold spot, GCc2, and GCc3 are never substituted for GCc1.
    """
    root = Path(input_dir)
    gold_path = root / "GCc1.csv"
    gold_export = _read_export(gold_path)
    close = _numeric_field(gold_export, "TRDPRC_1", gold_path)
    valid_close = close.notna()
    close_values = close.loc[valid_close].to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or (close_values <= 0).any():
        raise ValueError("GCc1 TRDPRC_1 prices must be finite and positive.")

    gold_prices = pd.DataFrame(
        {
            "date": gold_export.loc[valid_close, "Date"].to_numpy(),
            "close": close.loc[valid_close].to_numpy(),
        }
    )
    if "ACVOL_1" in gold_export.columns:
        volume = _numeric_field(gold_export, "ACVOL_1", gold_path)
        valid_volume = volume.dropna().to_numpy(dtype=float)
        if not np.isfinite(valid_volume).all() or (valid_volume < 0).any():
            raise ValueError("GCc1 ACVOL_1 volume must be finite and non-negative.")
        gold_prices["volume"] = volume.loc[valid_close].to_numpy()
    gold_prices.attrs.update(
        source="LSEG Workspace",
        instrument="COMEX Gold Futures generic first contract (GCc1)",
        currency="USD",
        price_unit="Verify in LSEG Workspace",
    )

    predictor_series: list[pd.DataFrame] = []
    sources: dict[str, str] = {}
    missing_files: list[str] = []
    series_specs = {
        **_PRICE_SERIES,
        **_OPTION_SERIES,
        **_OPTIONAL_SERIES,
    }
    for filename, specification in series_specs.items():
        feature, fields, description = specification
        series, source = _make_predictor_series(
            root, filename, feature, fields, description
        )
        if series is None:
            missing_files.append(filename)
            continue
        predictor_series.append(series)
        sources[feature] = source or ""

    sp500 = next(
        (series for series in predictor_series if "sp500_close" in series.columns),
        None,
    )
    if sp500 is not None:
        returns = sp500[["date", "sp500_close_observation_date"]].copy()
        returns["sp500_log_return"] = np.log(sp500["sp500_close"]).diff()
        returns["sp500_log_return_observation_date"] = returns[
            "sp500_close_observation_date"
        ]
        returns = returns.dropna(subset=["sp500_log_return"]).drop(
            columns="sp500_close_observation_date"
        )
        predictor_series.append(returns)
        sources["sp500_log_return"] = (
            "Close-to-close log difference derived from LSEG Workspace SPX.csv"
        )

    if not predictor_series:
        raise ValueError(
            f"No supported LSEG predictor CSVs were found in {root}."
        )

    predictors = predictor_series[0]
    for series in predictor_series[1:]:
        predictors = predictors.merge(
            series, on="date", how="outer", validate="one_to_one"
        )
    predictors = predictors.sort_values("date").reset_index(drop=True)
    feature_columns = [column for column in sources if column in predictors.columns]
    observation_columns = [
        f"{feature}_observation_date" for feature in feature_columns
    ]
    predictors[feature_columns + observation_columns] = predictors[
        feature_columns + observation_columns
    ].ffill()

    source_manifest = _read_source_manifest(root)
    manifest_series = source_manifest.get("series", {})
    if not isinstance(manifest_series, dict):
        manifest_series = {}
    unavailable_series = {
        ric: str(entry.get("status", "unknown"))
        for ric, entry in manifest_series.items()
        if isinstance(entry, dict) and entry.get("status") != "saved"
    }
    predictors.attrs.update(
        source="LSEG Workspace CSV exports",
        sources=sources,
        lseg_availability_lag="one weekday after source observation date (proxy)",
        lseg_import={
            "input_dir": str(root),
            "gold_file": gold_path.name,
            "gold_rows_read": int(len(gold_export)),
            "gold_rows_with_close": int(len(gold_prices)),
            "gold_rows_dropped_missing_close": int((~valid_close).sum()),
            "gold_volume_available": "volume" in gold_prices.columns,
            "loaded_predictor_files": sorted(
                filename
                for filename in series_specs
                if (root / filename).is_file()
            ),
            "missing_predictor_files": sorted(missing_files),
            "source_manifest_statuses": unavailable_series,
        },
    )
    return gold_prices, predictors
