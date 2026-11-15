"""Calibrate the exploratory quantile GARCH at one dated forecast origin."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gold_vol_pipeline import fit_quantile_garch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/stage2_multivariate.csv")
    parser.add_argument("--origin", default="2026-11-03")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    panel = pd.read_csv(args.input, parse_dates=["date"]).sort_values("date")
    if panel["date"].isna().any() or panel["date"].duplicated().any():
        raise ValueError("Panel dates must be nonmissing and unique.")
    origin = pd.Timestamp(args.origin)
    if origin not in set(panel["date"]):
        raise ValueError("Origin must be present in the input panel.")
    history = panel.loc[panel["date"] <= origin, "log_return"]
    if not history.empty and pd.isna(history.iloc[0]):
        history = history.iloc[1:]
    if history.isna().any():
        raise ValueError("Returns cannot be missing after the first panel row.")
    fit = fit_quantile_garch(history, quantile=args.quantile)
    result = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "origin": origin.date().isoformat(),
        "quantile": args.quantile,
        "horizon": args.horizon,
        "simulations": args.simulations,
        "seed": args.seed,
        "fit": asdict(fit),
        "forecast": fit.forecast(
            horizon=args.horizon, simulations=args.simulations, seed=args.seed
        ),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
