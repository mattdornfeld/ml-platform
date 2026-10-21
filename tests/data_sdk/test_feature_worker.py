"""Unit coverage for the API-dispatched feature Spark worker."""

from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock, patch

from ml_platform.data_sdk.materializers._base import MaterializerInputPartition, MaterializerPartition
from ml_platform.data_sdk.materializers._feature import FeatureMaterializer
from ml_platform.data_sdk.transforms._types import DatasetDependency


class PlannedInputTest(unittest.TestCase):
    """Verify the worker reads the immutable Iceberg plan supplied by the API."""

    @patch("ml_platform.data_sdk.materializers._feature.F")
    def test_reads_each_parent_at_its_pinned_snapshot(self, functions: MagicMock) -> None:
        """A newer table head cannot replace the parent snapshot selected by the API."""
        spark = MagicMock()
        frame = spark.read.option.return_value.table.return_value.where.return_value
        frame.dropDuplicates.return_value = frame
        callbacks = [MaterializerPartition(partition_id="output", timestamp=datetime(2026, 10, 19, tzinfo=timezone.utc), callback_token="token", dataset_partitions=[MaterializerInputPartition(partition_id="parent", timestamp=datetime(2026, 10, 19, tzinfo=timezone.utc), snapshot_id=42)])]

        result = FeatureMaterializer._planned_market_data(spark, callbacks, DatasetDependency(name="sp500", version=3, num_lookback_partitions=0))

        spark.read.option.assert_called_once_with("versionAsOf", 42)
        spark.read.option.return_value.table.assert_called_once_with("ml_platform.market_data.sp500-v3")
        self.assertIs(result, frame)

    def test_rejects_callback_without_a_parent_plan(self) -> None:
        """Workers fail closed instead of reading an unpinned table head."""
        with self.assertRaisesRegex(ValueError, "immutable dataset partition plan"):
            FeatureMaterializer._planned_market_data(MagicMock(), [], DatasetDependency(name="sp500", version=3, num_lookback_partitions=0))
