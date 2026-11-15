"""Screen 11 LSEG predictors against next-valid Yang-Zhang volatility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gold_vol_pipeline.lseg import load_lseg_workspace_data
from gold_vol_pipeline.yang_zhang import add_chronological_split, prepare_yang_zhang_dataset
from gold_vol_pipeline.yang_zhang_significance import (
    build_training_screen,
    select_predictors_by_vif,
    test_predictor_significance,
)

INPUT_FILES = (
    "GCc1.csv", "DXY.csv", "SPX.csv", "VIX.csv", "US10YT_RR.csv",
    "XAU1MO_R.csv", "XAU2MO_R.csv", "XAU3MO_R.csv", "XAU6MO_R.csv",
    "XAU1YO_R.csv", "XAU1MRR.csv", "XAU1MBF.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/refinitiv"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/stage1_predictor_significance"),
    )
    args = parser.parse_args()
    for name in INPUT_FILES:
        if not (args.input_dir / name).is_file():
            raise FileNotFoundError(f"Required LSEG source missing: {args.input_dir / name}")
    targets = add_chronological_split(
        prepare_yang_zhang_dataset(args.input_dir / "GCc1.csv", window=2)
    )
    _, predictors = load_lseg_workspace_data(args.input_dir)
    screen, metadata = build_training_screen(targets, predictors)
    retained, vif_history = select_predictors_by_vif(screen)
    results, model_metadata = test_predictor_significance(screen, predictors=retained)
    metadata.update(model_metadata)
    metadata.update({
        "vif_threshold": 5.0,
        "vif_selection": "remove highest-VIF candidate iteratively; always retain current-volatility control",
        "vif_removed_predictors": vif_history.loc[vif_history["removed"], "variable"].tolist(),
        "vif_final_max": float(vif_history.loc[
            vif_history["iteration"].eq(vif_history["iteration"].max()), "vif"
        ].max()),
    })
    metadata["input_files"] = {
        name: {
            "path": str(args.input_dir / name),
            "sha256": hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest(),
        }
        for name in INPUT_FILES
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen.to_csv(args.output_dir / "training_pairs.csv", index=False)
    vif_history.to_csv(args.output_dir / "vif_history.csv", index=False)
    results.to_csv(args.output_dir / "predictor_significance.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "input_files"}, indent=2))
    print(results.to_string(index=False))
    print("\nVIF selection history:")
    print(vif_history.to_string(index=False))
    print(f"Saved outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
