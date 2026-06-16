"""Build voyage snapshots with configurable time slice granularity.

This unified script replaces the separate monthly/biweekly snapshot generators.
Use the ``--slice`` argument to control the time window size:
  - ``monthly``: one snapshot per calendar month (default)
  - ``biweekly``: one snapshot every 14 days
  - ``weekly``: one snapshot every 7 days
  - ``<N>d``: custom interval of N days (e.g. ``10d`` for 10-day slices)

Optionally attach port coordinates via ``--with-coords`` when the source CSV
contains latitude/longitude columns for start and end ports.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Iterator

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CSV_PATH = BASE_DIR / "data" / "raw_data.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "snapshots"
DEFAULT_SAMPLE_OUTPUT_DIR = BASE_DIR / "data" / "snapshots_sampled"

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_SAMPLE_RATE = 0.01
DEFAULT_SLICE = "monthly"

START_PORT_LAT_COLUMN = "start_port_lat"
START_PORT_LON_COLUMN = "start_port_lon"
END_PORT_LAT_COLUMN = "end_port_lat"
END_PORT_LON_COLUMN = "end_port_lon"


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate voyage snapshots with configurable time slices"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to AIS training CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store snapshot CSVs",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=DEFAULT_SLICE,
        help="Time slice granularity: 'monthly', 'biweekly', 'weekly', or '<N>d' for N-day intervals",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="First date (YYYY-MM-DD) to consider, inclusive",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional last date (YYYY-MM-DD); defaults to final arrival date",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Skip slices with fewer than this many voyages",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help="Random sample rate per slice (0 disables sampling)",
    )
    parser.add_argument(
        "--sample-output-dir",
        type=Path,
        default=DEFAULT_SAMPLE_OUTPUT_DIR,
        help="Directory to store sampled snapshot CSVs",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility",
    )
    parser.add_argument(
        "--no-coords",
        action="store_true",
        help="Exclude port coordinates from output (default includes them)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Slice interval parsing
# ─────────────────────────────────────────────────────────────────────────────
SLICE_PRESETS = {
    "monthly": None,  # special handling
    "biweekly": 14,
    "weekly": 7,
}


def parse_slice_days(slice_str: str) -> int | None:
    """Return number of days for slice, or None for monthly mode."""
    lower = slice_str.lower().strip()
    if lower in SLICE_PRESETS:
        return SLICE_PRESETS[lower]

    match = re.fullmatch(r"(\d+)d", lower)
    if match:
        return int(match.group(1))

    raise ValueError(
        f"Invalid --slice value '{slice_str}'. "
        "Use 'monthly', 'biweekly', 'weekly', or '<N>d' (e.g. '10d')."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logging.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    for col in ("leg_start_postime", "arrival_time"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["leg_start_postime", "arrival_time"])
    logging.debug("Dataset size after dropping null timestamps: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate column validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_coordinate_columns(df: pd.DataFrame) -> None:
    required_columns = [
        START_PORT_LAT_COLUMN,
        START_PORT_LON_COLUMN,
        END_PORT_LAT_COLUMN,
        END_PORT_LON_COLUMN,
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required coordinate columns: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# Slice iteration
# ─────────────────────────────────────────────────────────────────────────────
def monthly_span(
    start: pd.Timestamp, end: pd.Timestamp
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Yield (slice_start, slice_end, label) for each calendar month."""
    current = start.replace(day=1)
    while current <= end:
        next_month = current + pd.offsets.MonthBegin(1)
        label = current.strftime("%Y-%m")
        yield current, next_month, label
        current = next_month


def fixed_day_span(
    start: pd.Timestamp, end: pd.Timestamp, days: int
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Yield (slice_start, slice_end, label) for fixed-day intervals."""
    current = start.normalize()
    while current <= end:
        slice_end = current + pd.Timedelta(days=days)
        label = current.strftime("%Y-%m-%d")
        yield current, slice_end, label
        current = slice_end


def slice_span(
    start: pd.Timestamp, end: pd.Timestamp, slice_days: int | None
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Unified iterator dispatching monthly or fixed-day mode."""
    if slice_days is None:
        yield from monthly_span(start, end)
    else:
        yield from fixed_day_span(start, end, slice_days)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot building
# ─────────────────────────────────────────────────────────────────────────────
def build_snapshots(
    df: pd.DataFrame,
    start_date: str,
    end_date: str | None,
    output_dir: Path,
    slice_days: int | None,
    include_coords: bool = True,
    min_rows: int = 1,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    sample_output_dir: Path | None = None,
    random_seed: int | None = None,
) -> tuple[list[Path], list[Path]]:
    start_ts = pd.to_datetime(start_date).normalize()
    if slice_days is None:
        start_ts = start_ts.replace(day=1)

    max_arrival = df["arrival_time"].max().normalize()
    if slice_days is None:
        max_arrival = max_arrival.replace(day=1)

    end_ts = pd.to_datetime(end_date).normalize() if end_date else max_arrival

    if end_ts < start_ts:
        raise ValueError("end_date precedes start_date")

    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    logging.info("Cleared output directory %s", output_dir)
    sample_rate = max(0.0, sample_rate)
    if sample_rate > 0 and sample_output_dir is not None:
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        for item in sample_output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logging.info("Cleared sample output directory %s", sample_output_dir)

    exported_files: list[Path] = []
    sampled_files: list[Path] = []

    for slice_start, slice_end, label in slice_span(start_ts, end_ts, slice_days):
        mask = (
            (df["leg_start_postime"] >= slice_start)
            & (df["leg_start_postime"] < slice_end)
        ) | (
            (df["arrival_time"] >= slice_start) & (df["arrival_time"] < slice_end)
        )
        slice_df = df.loc[mask].copy()

        if slice_df.empty or len(slice_df) < min_rows:
            logging.debug("Skipping %s (%d rows)", label, len(slice_df))
            continue

        slice_df.sort_values(["arrival_time", "leg_start_postime"], inplace=True)

        # Build edge DataFrame
        if include_coords:
            export_columns = [
                "leg_start_port_code",
                "leg_end_port_code",
                "dwt",
                START_PORT_LAT_COLUMN,
                START_PORT_LON_COLUMN,
                END_PORT_LAT_COLUMN,
                END_PORT_LON_COLUMN,
            ]
            rename_map = {
                "leg_start_port_code": "u",
                "leg_end_port_code": "v",
                "dwt": "w",
                START_PORT_LAT_COLUMN: "u_lat",
                START_PORT_LON_COLUMN: "u_lon",
                END_PORT_LAT_COLUMN: "v_lat",
                END_PORT_LON_COLUMN: "v_lon",
            }
            dropna_cols = [
                "u",
                "v",
                "w",
                "u_lat",
                "u_lon",
                "v_lat",
                "v_lon",
            ]
        else:
            export_columns = ["leg_start_port_code", "leg_end_port_code", "dwt"]
            rename_map = {
                "leg_start_port_code": "u",
                "leg_end_port_code": "v",
                "dwt": "w",
            }
            dropna_cols = ["u", "v", "w"]

        edges_df = (
            slice_df[export_columns].rename(columns=rename_map).dropna(subset=dropna_cols)
        )
        if edges_df.empty:
            logging.debug("Skipping %s after edge projection (0 edges)", label)
            continue

        out_path = output_dir / f"snapshot_{label}.csv"
        edges_df.to_csv(out_path, index=False)
        logging.info("Exported %s edges for %s to %s", len(edges_df), label, out_path)
        exported_files.append(out_path)

        if sample_rate > 0 and sample_output_dir is not None:
            sample_count = max(1, int(round(len(edges_df) * sample_rate)))
            sample_count = min(sample_count, len(edges_df))
            sample_df = edges_df.sample(
                n=sample_count, random_state=random_seed, replace=False
            )
            sample_path = sample_output_dir / f"snapshot_{label}_sampled.csv"
            sample_df.to_csv(sample_path, index=False)
            logging.info(
                "Exported sampled snapshot (%s edges, rate %.2f%%) for %s to %s",
                len(sample_df),
                sample_rate * 100,
                label,
                sample_path,
            )
            sampled_files.append(sample_path)

    return exported_files, sampled_files


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    slice_days = parse_slice_days(args.slice)
    slice_desc = f"{slice_days}-day" if slice_days else "monthly"
    logging.info("Slice mode: %s", slice_desc)

    df = load_dataset(args.csv_path)

    include_coords = not args.no_coords
    if include_coords:
        validate_coordinate_columns(df)

    exported_files, sampled_files = build_snapshots(
        df,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        slice_days=slice_days,
        include_coords=include_coords,
        min_rows=args.min_rows,
        sample_rate=args.sample_rate,
        sample_output_dir=args.sample_output_dir,
        random_seed=args.random_seed,
    )

    if not exported_files:
        logging.warning("No snapshots met the export criteria")
    else:
        logging.info("Finished exporting %d snapshots", len(exported_files))
    if sampled_files:
        logging.info("Finished exporting %d sampled snapshots", len(sampled_files))


if __name__ == "__main__":
    main()
