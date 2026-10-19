"""Open a pandas-backed IPython session for a directory of Parquet files."""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from IPython import embed


def load_partitioned_parquet(directory: str | Path) -> pd.DataFrame:
    """Load every Parquet file below ``directory`` into one pandas DataFrame.

    The returned frame contains only columns stored in the Parquet files. The
    directory must contain at least one Parquet file.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Parquet directory does not exist: {root}")

    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet files found below: {root}")

    dataset = ds.dataset(parquet_files, format="parquet")
    return dataset.to_table().to_pandas()


def parse_args() -> argparse.Namespace:
    """Parse the directory containing Parquet data files to explore."""
    parser = argparse.ArgumentParser(
        description="Load partitioned Parquet files into pandas and open IPython."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Table or data directory containing Parquet files recursively.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the requested table and open an IPython REPL with it in scope."""
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    frame = load_partitioned_parquet(directory)
    parquet_files = sorted(directory.rglob("*.parquet"))
    print(f"Loaded {len(frame):,} rows from {len(parquet_files):,} Parquet files in {directory}")
    embed(
        header="Parquet explorer: `frame` and `data` contain the loaded pandas DataFrame.",
        user_ns={
            "data": frame,
            "directory": directory,
            "frame": frame,
            "parquet_files": parquet_files,
            "pd": pd,
        },
    )


if __name__ == "__main__":
    main()
