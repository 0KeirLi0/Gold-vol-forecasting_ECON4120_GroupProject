"""Command-line entry point for the gold futures data pipeline."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .pipeline import (
    DEFAULT_HORIZON,
    DEFAULT_TARGET,
    YAHOO_TICKER,
    download_futures_prices,
    load_gold_prices_csv,
    prepare_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare Yahoo COMEX gold futures prices."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional input CSV; by default data are downloaded from Yahoo Finance.",
    )
    parser.add_argument(
        "--start",
        help="First download date (default: 2015-01-01).",
    )
    parser.add_argument(
        "--end",
        help="Exclusive download end date (default: today, excluding today's partial bar).",
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
        default="Gold futures (specify contract or series)",
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
        help="Prepared output CSV (default: data/processed/gold_futures.csv).",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Path for raw downloaded prices (default: data/raw/gold_futures.csv).",
    )
    args = parser.parse_args()

    if args.input is not None:
        if args.end is not None or args.start is not None:
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
        end = args.end or date.today().isoformat()
        prices = download_futures_prices(args.start or "2015-01-01", end)
    dataset = prepare_dataset(prices, horizon=args.horizon, target=args.target)
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
