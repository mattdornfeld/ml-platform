"""Tests for local and MinIO Parquet explorer source resolution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ml_platform.data_sdk.utils.data_explorer import (
    ParquetExplorerSourceError,
    _parquet_source,
    load_partitioned_parquet,
)


class DataExplorerTest(unittest.TestCase):
    """Exercise source validation and local partitioned reads."""

    def test_loads_every_local_parquet_file(self) -> None:
        """Local recursive discovery combines all materialized partitions."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "timestamp_day=2026-01-01"
            second = root / "timestamp_day=2026-01-02"
            first.mkdir()
            second.mkdir()
            pd.DataFrame({"value": [1]}).to_parquet(first / "part.parquet")
            pd.DataFrame({"value": [2]}).to_parquet(second / "part.parquet")
            frame = load_partitioned_parquet(root)
        self.assertEqual(sorted(frame["value"].tolist()), [1, 2])

    def test_rejects_missing_or_empty_local_directories(self) -> None:
        """Explorer errors clearly before creating an invalid PyArrow dataset."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ParquetExplorerSourceError):
                _parquet_source(Path(temporary_directory) / "missing")
            with self.assertRaises(ParquetExplorerSourceError):
                _parquet_source(temporary_directory)

    @patch("ml_platform.data_sdk.utils.data_explorer._minio_filesystem")
    def test_s3_uri_uses_minio_filesystem(self, filesystem: object) -> None:
        """S3A table URIs become bucket-qualified PyArrow paths."""
        source, resolved_filesystem = _parquet_source(
            "s3a://warehouse/features/foo"
        )
        self.assertEqual(source, "warehouse/features/foo")
        self.assertIs(resolved_filesystem, filesystem.return_value)
