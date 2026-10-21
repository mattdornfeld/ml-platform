"""Unit coverage for the API-dispatched feature Spark worker."""

import unittest
from unittest.mock import MagicMock, patch

from ml_platform.data_sdk.feature import _planned_market_data


class PlannedInputTest(unittest.TestCase):
    """Verify the worker reads the immutable Iceberg plan supplied by the API."""

    @patch("ml_platform.data_sdk.feature.F")
    def test_reads_each_parent_at_its_pinned_snapshot(self, functions: MagicMock) -> None:
        """A newer table head cannot replace the parent snapshot selected by the API."""
        spark = MagicMock()
        frame = spark.read.option.return_value.table.return_value.where.return_value
        frame.dropDuplicates.return_value = frame
        callbacks = [{"dataset_partitions": [{"partition_id": "parent", "timestamp": "2026-10-19T00:00:00Z", "snapshot_id": 42}]}]

        result = _planned_market_data(spark, callbacks)

        spark.read.option.assert_called_once_with("versionAsOf", 42)
        spark.read.option.return_value.table.assert_called_once_with("data.market_data.hourly_ohlcv")
        self.assertIs(result, frame)

    def test_rejects_callback_without_a_parent_plan(self) -> None:
        """Workers fail closed instead of reading an unpinned table head."""
        with self.assertRaisesRegex(ValueError, "immutable dataset partition plan"):
            _planned_market_data(MagicMock(), [{"dataset_partitions": []}])
