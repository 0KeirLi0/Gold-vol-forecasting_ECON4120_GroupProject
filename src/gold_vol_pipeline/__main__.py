"""Command-line entry point for the gold spot data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    DEFAULT_HORIZON,
    DEFAULT_TARGET,
    load_spot_prices_csv,
    prepare_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare daily gold spot prices and a forward volatility target."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file containing one consistent gold spot-price series.",
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
        default="Gold spot",
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
        default=Path("data/processed/gold_spot.csv"),
        help="Output CSV path (default: data/processed/gold_spot.csv).",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional path for a normalized raw-price copy. The input file is "
        "never modified or copied by default.",
    )
    args = parser.parse_args()

    prices = load_spot_prices_csv(
        args.input,
        date_column=args.date_column,
        close_column=args.close_column,
        source=args.source,
        instrument=args.instrument,
        currency=args.currency,
        price_unit=args.price_unit,
    )
    dataset = prepare_dataset(prices, horizon=args.horizon, target=args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.raw_output
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
