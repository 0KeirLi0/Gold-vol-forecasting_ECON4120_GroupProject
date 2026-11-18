"""Download gold and candidate predictor histories from LSEG Workspace.

Run this on a computer with LSEG Workspace installed, signed in, and open.
RICs and fields in PROJECT_PANEL_REQUESTS are starting mappings: verify them
in Workspace before interpreting or modeling their values.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# User-supplied gold futures, spot/fixing, and implied-volatility examples.
EXAMPLE_REQUESTS: dict[str, list[str]] = {
    "GCc1": ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_1", "OPINT_1"],
    "GCc2": ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_1", "OPINT_1"],
    "GCc3": ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_1", "OPINT_1"],
    "XAUFIX=": ["TRDPRC_1"],
    "XAU=": ["TRDPRC_1"],
    "XAU1MO=R": ["MID_PRICE"],
    "XAU2MO=R": ["MID_PRICE"],
    "XAU3MO=R": ["MID_PRICE"],
    "XAU6MO=R": ["MID_PRICE"],
    "XAU1YO=R": ["MID_PRICE"],
    "XAU1MRR=": ["MID_PRICE"],
    "XAU1MBF=": ["MID_PRICE"],
}

# Candidate mappings for the project's predictor variables. These RICs
# and fields are not confirmed project decisions; verify availability and units
# in Workspace. Change them here if your Workspace uses different instruments.
PROJECT_PANEL_REQUESTS: dict[str, list[str]] = {
    ".DXY": ["TRDPRC_1"],  # USD strength
    ".SPX": ["TRDPRC_1"],  # S&P 500 price index; derive log returns later
    ".VIX": ["TRDPRC_1"],  # VIX
    "US10YT=RR": ["BID", "ASK"],  # 10-year US Treasury yield; derive midpoint
    "USBEI10Y=RR": ["BID", "ASK"],  # Candidate breakeven RIC; verify
    "USCRWT=ECI": ["TRDPRC_1"],  # Candidate WTI spot; verify definition
}


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "series"


def _sheet_name(value: str, used: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", value)[:31] or "series"
    name = base
    suffix = 2
    while name.casefold() in used:
        ending = f"_{suffix}"
        name = f"{base[: 31 - len(ending)]}{ending}"
        suffix += 1
    used.add(name.casefold())
    return name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="First history date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="Last history date (YYYY-MM-DD); defaults to yesterday",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/refinitiv"),
        help="Output directory (default: data/refinitiv)",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Download only the supplied gold futures, spot, and volatility examples",
    )
    args = parser.parse_args()
    if date.fromisoformat(args.start) > date.fromisoformat(args.end):
        parser.error("--start must be on or before --end")
    return args


def main() -> int:
    args = _parse_args()
    requests: dict[str, tuple[list[str], str]] = {
        ric: (fields, "gold_examples")
        for ric, fields in EXAMPLE_REQUESTS.items()
    }

    if not args.examples_only:
        for ric, fields in PROJECT_PANEL_REQUESTS.items():
            requests[ric] = (fields, "stage2_candidates")
        # GCc1 is already in EXAMPLE_REQUESTS and supplies candidate gold volume.
        requests["GCc1"] = (
            EXAMPLE_REQUESTS["GCc1"],
            "gold_examples_and_stage2_volume",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    manifest: dict[str, Any] = {
        "provider": "LSEG Workspace",
        "start": args.start,
        "end": args.end,
        "end_date_note": (
            "The requested end date may have no observation on non-trading days."
        ),
        "availability_note": (
            "Files contain source observation dates, not reconstructed historical "
            "publication timestamps. Do not assume same-day predictor availability."
        ),
        "series": {},
    }

    import lseg.data as ld

    print("Connecting to LSEG Workspace...")
    session_open = False
    try:
        ld.open_session()
        session_open = True
        for ric, (fields, category) in requests.items():
            print(f"Downloading {ric} ({', '.join(fields)}) ...")
            entry: dict[str, Any] = {
                "category": category,
                "fields": fields,
                "status": "error",
            }
            try:
                frame = ld.get_history(
                    universe=ric,
                    fields=fields,
                    start=args.start,
                    end=args.end,
                )
                if frame is None or frame.empty:
                    entry["status"] = "no_data"
                    entry["message"] = "Workspace returned no observations."
                    print("  -> No data returned")
                    manifest["series"][ric] = entry
                    continue

                csv_path = args.output_dir / f"{_safe_filename(ric)}.csv"
                frame.to_csv(csv_path, index=True, index_label="Date")
                results[ric] = frame
                entry.update(
                    {
                        "status": "saved",
                        "rows": int(len(frame)),
                        "columns": [str(column) for column in frame.columns],
                        "file": csv_path.name,
                    }
                )
                print(f"  -> Saved {csv_path} ({len(frame)} rows)")
            except Exception as error:
                entry["message"] = f"{type(error).__name__}: {error}"
                print(f"  -> Error: {entry['message']}")
            manifest["series"][ric] = entry
    finally:
        if session_open:
            ld.close_session()

    if results:
        workbook_path = args.output_dir / "ALL_LSEG_DATA.xlsx"
        used_sheet_names: set[str] = set()
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for ric, frame in results.items():
                frame.to_excel(
                    writer,
                    sheet_name=_sheet_name(ric, used_sheet_names),
                    index=True,
                index_label="Date",
                )
        print(f"Combined workbook saved: {workbook_path}")

    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest saved: {manifest_path}")
    print(
        f"Download finished: {len(results)} of {len(requests)} series saved. "
        f"Files are in: {args.output_dir.resolve()}"
    )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
