"""Tests for loading a partitioned Parquet directory into pandas."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from ml_platform.sdk.parquet_explorer import load_partitioned_parquet


class ParquetExplorerTest(unittest.TestCase):
    """Verify the explorer reads nested Parquet files without path metadata."""

    def test_loads_nested_parquet_files_without_hive_partition_columns(self) -> None:
        """Combine every data file without adding directory names as columns."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "timestamp_day=2026-01-02"
            second = root / "timestamp_day=2026-01-03"
            first.mkdir()
            second.mkdir()
            pd.DataFrame({"symbol": ["AAA"], "Close": [100.0]}).to_parquet(
                first / "first.parquet", index=False
            )
            pd.DataFrame({"symbol": ["BBB"], "Close": [200.0]}).to_parquet(
                second / "second.parquet", index=False
            )

            frame = load_partitioned_parquet(root)

        self.assertEqual(set(frame["symbol"]), {"AAA", "BBB"})
        self.assertNotIn("timestamp_day", frame.columns)

    def test_rejects_directory_without_parquet_files(self) -> None:
        """Report an actionable error rather than opening an empty shell."""
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "No Parquet files found"):
                load_partitioned_parquet(temporary_directory)


if __name__ == "__main__":
    unittest.main()
