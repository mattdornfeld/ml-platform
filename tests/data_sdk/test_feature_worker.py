"""Unit coverage for the API-dispatched feature Spark worker."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
import unittest
from unittest.mock import MagicMock, patch

from pyspark.sql import DataFrame

from ml_platform.data_sdk.materializers._base import MaterializerInputPartition, MaterializerPartition
from ml_platform.data_sdk.materializers._feature import FeatureMaterializationConfig, FeatureMaterializer
from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._types import Dataset, FeatureDatasetDependency


@dataclass(frozen=True)
class TwoInputTransform(FeatureTransform):
    """Test transform whose constructor requires two independent datasets."""

    prices: Dataset
    fundamentals: Dataset
    partition_type: Literal["hourly", "daily"]

    def __call__(self) -> DataFrame:
        return self.prices.frame


class PlannedInputTest(unittest.TestCase):
    """Verify the worker reads the immutable Iceberg plan supplied by the API."""

    @patch("ml_platform.data_sdk.materializers._feature.F")
    def test_reads_each_parent_at_its_pinned_snapshot(self, functions: MagicMock) -> None:
        """A newer table head cannot replace the parent snapshot selected by the API."""
        spark = MagicMock()
        frame = spark.read.option.return_value.table.return_value.where.return_value
        frame.dropDuplicates.return_value = frame
        callbacks = [MaterializerPartition(partition_id="output", timestamp=datetime(2026, 10, 19, tzinfo=timezone.utc), callback_token="token", dataset_partitions=[
            MaterializerInputPartition(partition_id="parent", dataset_name="sp500", dataset_version=3, timestamp=datetime(2026, 10, 19, tzinfo=timezone.utc), snapshot_id=42),
            MaterializerInputPartition(partition_id="other-parent", dataset_name="company_facts", dataset_version=2, timestamp=datetime(2026, 10, 19, tzinfo=timezone.utc), snapshot_id=99),
        ])]

        result = FeatureMaterializer._planned_dataset(spark, callbacks, FeatureDatasetDependency(name="sp500", version=3, argument_name="market_data", num_lookback_partitions=0))

        spark.read.option.assert_called_once_with("versionAsOf", 42)
        spark.read.option.return_value.table.assert_called_once_with("ml_platform.market_data.`sp500-v3`")
        self.assertIs(result, frame)

    def test_rejects_callback_without_a_parent_plan(self) -> None:
        """Workers fail closed instead of reading an unpinned table head."""
        with self.assertRaisesRegex(ValueError, "immutable dataset partition plan"):
            FeatureMaterializer._planned_dataset(MagicMock(), [], FeatureDatasetDependency(name="sp500", version=3, argument_name="market_data", num_lookback_partitions=0))

    @patch.object(FeatureMaterializer, "_planned_dataset")
    def test_injects_each_declared_dataset_under_its_argument_name(self, planned_dataset: MagicMock) -> None:
        """Multiple datasets retain their own identity, snapshot plan, and constructor slot."""
        timestamp = datetime(2026, 10, 19, tzinfo=timezone.utc)
        config = FeatureMaterializationConfig.model_validate({
            "name": "joined_feature", "version": 1, "description": "Joined feature.",
            "warehouse_dir": "temporary-warehouse", "dtype": "double", "frequency": "daily",
            "transform": "ml_platform.data_sdk.feature_transforms._joined.JoinedTransform",
            "datasets": [
                {"name": "prices_v2", "version": 1, "argument_name": "prices", "num_lookback_partitions": 1},
                {"name": "company_facts", "version": 2, "argument_name": "fundamentals", "num_lookback_partitions": 0},
            ],
            "partitions": [{
                "partition_id": "output", "timestamp": timestamp, "callback_token": "token",
                "dataset_partitions": [
                    {"partition_id": "prices-parent", "dataset_name": "prices_v2", "dataset_version": 1,
                     "timestamp": timestamp, "snapshot_id": 42},
                    {"partition_id": "facts-parent", "dataset_name": "company_facts", "dataset_version": 2,
                     "timestamp": timestamp, "snapshot_id": 99},
                ],
            }],
        })
        price_frame, facts_frame = MagicMock(), MagicMock()
        planned_dataset.side_effect = [price_frame, facts_frame]
        worker = FeatureMaterializer(config)

        datasets = worker._load_planned_datasets(MagicMock())
        with patch.object(type(worker), "_load_transform", return_value=TwoInputTransform):
            feature = worker._feature_transform(datasets)

        self.assertEqual(set(datasets), {"prices", "fundamentals"})
        self.assertIs(feature, price_frame)
        self.assertIs(datasets["fundamentals"].frame, facts_frame)
        self.assertEqual(planned_dataset.call_args_list[0].args[2].name, "prices_v2")
        self.assertEqual(planned_dataset.call_args_list[1].args[2].name, "company_facts")
