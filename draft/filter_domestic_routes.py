"""
Filter out domestic shipping routes from a raw CSV dataset.

Definition: A route is considered domestic if the first two letters
of the origin port code (`leg_start_port_code`) and the destination
port code (`leg_end_port_code`) are the same (e.g., CN→CN, US→US).

Usage:
  python3 scripts/filter_domestic_routes.py \
      --input data/raw_data.csv \
      --output data/raw_data_international.csv

Notes:
- Keeps rows where country code cannot be determined (missing/short codes).
- Streams line-by-line using the csv module (no external deps).
"""

import argparse
import csv
import os
from typing import Tuple

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data"

DEFAULT_INPUT = DATA_DIR / "raw_data.csv"
DEFAULT_OUTPUT = DATA_DIR / "raw_data_international.csv"

def same_country_prefix(a: str, b: str) -> bool:
    """Return True if both strings exist and their first two letters match.

    Empty or too-short values will return False (treated as unknown, kept).
    """
    if not a or not b:
        return False
    a = a.strip()
    b = b.strip()
    if len(a) < 2 or len(b) < 2:
        return False
    return a[:2].upper() == b[:2].upper()


def filter_csv(input_path: str, output_path: str) -> Tuple[int, int]:
    """Filter input CSV and write only international routes to output.

    Returns a tuple of (kept_rows, dropped_rows).
    """
    kept = 0
    dropped = 0

    with open(input_path, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV has no header.")

        required_cols = ["leg_start_port_code", "leg_end_port_code"]
        for col in required_cols:
            if col not in fieldnames:
                raise ValueError(
                    f"Missing required column '{col}' in input CSV. Found: {fieldnames}"
                )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                start_code = row.get("leg_start_port_code", "")
                end_code = row.get("leg_end_port_code", "")

                if same_country_prefix(start_code, end_code):
                    dropped += 1
                    continue

                writer.writerow(row)
                kept += 1

    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove domestic routes (same first two letters of start/end port codes) "
            "from a raw shipping dataset."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()

    kept, dropped = filter_csv(args.input, args.output)

    print(
        "Filtering complete.\n"
        f"Input: {args.input}\n"
        f"Output: {args.output}\n"
        f"Kept (international): {kept}\n"
        f"Dropped (domestic): {dropped}"
    )


if __name__ == "__main__":
    main()
