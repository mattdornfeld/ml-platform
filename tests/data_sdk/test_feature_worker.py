"""Unit coverage for the API-dispatched feature Spark worker."""

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from pyspark.sql import DataFrame

from ml_platform.common.domain import FeatureInputWindow
from ml_platform.common.types import DatasetType, DataType, RunFrequency
from ml_platform.data_sdk import Dataset, FeatureTransform, LabelTransform
from ml_platform.data_sdk.exceptions import DatasetPartitionPlanError
from ml_platform.data_sdk.label_transforms import (
    TripleBarrierLabelTransform,
)
from ml_platform.data_sdk.materializers import (
    FeatureMaterializationConfig,
    FeatureMaterializer,
    FeatureMaterializerDatasetDependency,
    LabelMaterializationConfig,
    LabelMaterializer,
    MaterializerPartition,
    PartitionDependency,
)
from ml_platform.data_sdk.utils import record


@record
class TwoInputTransform(FeatureTransform):
    """Test transform whose constructor requires two independent datasets."""

    prices: Dataset
    fundamentals: Dataset

    def __call__(self) -> DataFrame:
        return self.prices.frame


class PlannedInputTest(unittest.TestCase):
    """Verify the worker reads the immutable Iceberg plan supplied by the API."""

    @patch(
        "ml_platform.data_sdk.materializers._transform_materializer.spark_functions"
    )
    def test_reads_each_parent_at_its_pinned_snapshot(
        self, functions: MagicMock
    ) -> None:
        """A newer table head cannot replace the parent snapshot selected by the API."""
        spark = MagicMock()
        frame = (
            spark.read.option.return_value.table.return_value.where.return_value
        )
        frame.dropDuplicates.return_value = frame
        callbacks = [
            MaterializerPartition(
                partition_id="output",
                timestamp=datetime(2026, 10, 19, tzinfo=UTC),
                callback_token="token",
                partition_dependencies=[
                    PartitionDependency(
                        partition_id="parent",
                        dataset_name="sp500",
                        dataset_version=3,
                        timestamp=datetime(2026, 10, 19, tzinfo=UTC),
                        snapshot_id=42,
                    ),
                    PartitionDependency(
                        partition_id="other-parent",
                        dataset_name="company_facts",
                        dataset_version=2,
                        timestamp=datetime(2026, 10, 19, tzinfo=UTC),
                        snapshot_id=99,
                    ),
                ],
            )
        ]

        result = FeatureMaterializer._planned_dataset(
            spark,
            callbacks,
            FeatureMaterializerDatasetDependency(
                name="sp500",
                version=3,
                dataset_type=DatasetType.MARKET_DATA,
                input_window=FeatureInputWindow(lookback_partitions=0),
            ),
        )

        spark.read.option.assert_called_once_with("versionAsOf", 42)
        spark.read.option.return_value.table.assert_called_once_with(
            "ml_platform.market_data.`sp500-v3`"
        )
        self.assertIs(result, frame)

    def test_rejects_callback_without_a_parent_plan(self) -> None:
        """Workers fail closed instead of reading an unpinned table head."""
        with self.assertRaisesRegex(
            DatasetPartitionPlanError, "immutable partition plan"
        ):
            FeatureMaterializer._planned_dataset(
                MagicMock(),
                [],
                FeatureMaterializerDatasetDependency(
                    name="sp500",
                    version=3,
                    dataset_type=DatasetType.MARKET_DATA,
                    input_window=FeatureInputWindow(lookback_partitions=0),
                ),
            )

    @patch.object(FeatureMaterializer, "_planned_dataset")
    def test_injects_each_declared_dataset_under_its_dataset_name(
        self, planned_dataset: MagicMock
    ) -> None:
        """Multiple datasets retain their identity, snapshot plan, and name slot."""
        timestamp = datetime(2026, 10, 19, tzinfo=UTC)
        config = FeatureMaterializationConfig.model_validate(
            {
                "name": "joined_feature",
                "version": 1,
                "description": "Joined feature.",
                "environment": "local",
                "warehouse_dir": "temporary-warehouse",
                "dtype": DataType.DOUBLE,
                "frequency": RunFrequency.DAILY,
                "transform": "ml_platform.data_sdk.feature_transforms._joined.JoinedTransform",
                "datasets": [
                    {
                        "name": "prices",
                        "version": 1,
                        "dataset_type": DatasetType.MARKET_DATA,
                        "input_window": {"lookback_partitions": 1},
                    },
                    {
                        "name": "fundamentals",
                        "version": 2,
                        "dataset_type": DatasetType.MARKET_DATA,
                        "input_window": {"lookback_partitions": 0},
                    },
                ],
                "partitions": [
                    {
                        "partition_id": "output",
                        "timestamp": timestamp,
                        "callback_token": "token",
                        "partition_dependencies": [
                            {
                                "partition_id": "prices-parent",
                                "dataset_name": "prices",
                                "dataset_version": 1,
                                "timestamp": timestamp,
                                "snapshot_id": 42,
                            },
                            {
                                "partition_id": "facts-parent",
                                "dataset_name": "fundamentals",
                                "dataset_version": 2,
                                "timestamp": timestamp,
                                "snapshot_id": 99,
                            },
                        ],
                    }
                ],
            }
        )
        price_frame, facts_frame = MagicMock(), MagicMock()
        planned_dataset.side_effect = [price_frame, facts_frame]
        worker = FeatureMaterializer(config)

        datasets = worker._load_planned_datasets(MagicMock())
        feature = worker._inject_transform(TwoInputTransform, datasets)()

        self.assertEqual(set(datasets), {"prices", "fundamentals"})
        self.assertIs(feature, price_frame)
        self.assertIs(datasets["fundamentals"].frame, facts_frame)
        self.assertEqual(
            planned_dataset.call_args_list[0].args[2].name, "prices"
        )
        self.assertEqual(
            planned_dataset.call_args_list[1].args[2].name, "fundamentals"
        )


class TransformMaterializerTest(unittest.TestCase):
    """Verify feature and label workers use the same pinned-dataset mixin."""

    @patch.object(LabelMaterializer, "_planned_dataset")
    @patch.object(
        LabelMaterializer,
        "_load_transform",
        return_value=TripleBarrierLabelTransform,
    )
    def test_label_injects_pinned_datasets_through_the_shared_mixin(
        self,
        load_transform: MagicMock,
        planned_dataset: MagicMock,
    ) -> None:
        """Labels use the same dataset read and constructor injection contract as features."""
        timestamp = datetime(2026, 10, 19, tzinfo=UTC)
        config = LabelMaterializationConfig.model_validate(
            {
                "name": "triple_barrier",
                "version": 1,
                "description": "Labels.",
                "environment": "local",
                "warehouse_dir": "temporary-warehouse",
                "dtype": DataType.INT,
                "frequency": RunFrequency.DAILY,
                "keys": ["symbol"],
                "datasets": [
                    {
                        "name": "market_data",
                        "version": 3,
                        "dataset_type": DatasetType.MARKET_DATA,
                        "input_window": {
                            "lookback_partitions": 1,
                            "lookahead_partitions": 1,
                        },
                    }
                ],
                "transform": "ml_platform.data_sdk.label_transforms.TripleBarrierLabelTransform",
                "parameters": {
                    "horizon_bars": 24,
                    "volatility_lookback_bars": 24,
                    "profit_taking_multiplier": 1.0,
                    "stop_loss_multiplier": 1.0,
                },
                "partitions": [
                    {
                        "partition_id": "output",
                        "timestamp": timestamp,
                        "callback_token": "token",
                        "partition_dependencies": [
                            {
                                "partition_id": "market",
                                "dataset_name": "market_data",
                                "dataset_version": 3,
                                "timestamp": timestamp,
                                "snapshot_id": 42,
                            }
                        ],
                    }
                ],
            }
        )
        source_data_frame = MagicMock()
        planned_dataset.return_value = source_data_frame

        transform = LabelMaterializer(config)._transform(MagicMock())

        self.assertIs(transform.market_data.frame, source_data_frame)
        planned_dataset.assert_called_once()
        load_transform.assert_called_once_with(config.transform, LabelTransform)
