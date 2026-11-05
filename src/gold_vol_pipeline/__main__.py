"""Command-line entry point for the gold futures data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    DEFAULT_HORIZON,
    DEFAULT_TARGET,
    YAHOO_TICKER,
    download_gold_prices,
    load_gold_prices_csv,
    prepare_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare daily COMEX gold futures data from Yahoo Finance or CSV."
    )
    parser.add_argument("--start", help="First date, YYYY-MM-DD (Yahoo download).")
    parser.add_argument(
        "--end",
        help="Exclusive end date, YYYY-MM-DD (Yahoo download convention).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Use a local gold-price CSV instead of downloading from Yahoo.",
    )
    parser.add_argument(
        "--date-column",
        default="date",
        help="Date column in --input CSV (default: date).",
    )
    parser.add_argument(
        "--close-column",
        default="close",
        help="Closing-price column in --input CSV (default: close).",
    )
    parser.add_argument(
        "--source",
        default="User-supplied CSV",
        help="Source/provider label for --input data.",
    )
    parser.add_argument(
        "--instrument",
        default="COMEX Gold Futures (continuous series)",
        help="Instrument label for --input data.",
    )
    parser.add_argument(
        "--currency",
        default="USD",
        help="Price currency for --input data.",
    )
    parser.add_argument(
        "--price-unit",
        default="USD per troy ounce",
        help="Price units for --input data.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Number of following observed sessions in the target (default: {DEFAULT_HORIZON}).",
    )
    parser.add_argument(
        "--target",
        choices=("realized_variance", "realized_volatility"),
        default=DEFAULT_TARGET,
        help=f"Forward target measure (default: {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/gold_futures.csv"),
        help="Output CSV path (default: data/processed/gold_futures.csv).",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional path for a normalized raw-price copy. Yahoo downloads "
        "default to data/raw/gold_futures.csv; CSV input is not copied by default.",
    )
    args = parser.parse_args()

    if args.input is not None:
        if args.start is not None or args.end is not None:
            parser.error("--input cannot be combined with --start or --end.")
        prices = load_gold_prices_csv(
            args.input,
            date_column=args.date_column,
            close_column=args.close_column,
            source=args.source,
            instrument=args.instrument,
            currency=args.currency,
            price_unit=args.price_unit,
        )
    else:
        if args.start is None or args.end is None:
            parser.error("Provide --input or both --start and --end.")
        prices = download_gold_prices(args.start, args.end)
    dataset = prepare_dataset(prices, horizon=args.horizon, target=args.target)
    if "source" not in prices.attrs:
        prices.attrs.update(
            source="Yahoo Finance",
            instrument="COMEX Gold Futures (continuous series, GC=F)",
            currency="USD",
            price_unit="USD per troy ounce",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.raw_output
    if raw_output is None and args.input is None:
        raw_output = Path("data/raw/gold_futures.csv")
    if raw_output is not None:
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(raw_output, index=False)
    dataset.to_csv(args.output, index=False)
    message = (
        f"Wrote {len(dataset)} forecast rows to {args.output} "
        f"({prices.attrs['source']}, {prices.attrs['instrument']}, "
        f"horizon={args.horizon}, target={args.target})."
    )
    if raw_output is not None:
        message += f" Raw prices saved to {raw_output}."
    print(message)


if __name__ == "__main__":
    main()
