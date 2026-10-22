"""Unit coverage for the API-dispatched feature Spark worker."""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from pyspark.sql import DataFrame

from ml_platform.data_sdk.exceptions import DatasetPartitionPlanError
from ml_platform.data_sdk.label_transforms._triple_barrier import (
    TripleBarrierLabelTransform,
)
from ml_platform.data_sdk.materializers._base import (
    MaterializerPartition,
    PartitionDependency,
)
from ml_platform.data_sdk.materializers._feature import (
    FeatureMaterializationConfig,
    FeatureMaterializer,
)
from ml_platform.data_sdk.materializers._label import (
    LabelMaterializationConfig,
    LabelMaterializer,
)
from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._label_transform import LabelTransform
from ml_platform.data_sdk.transforms._types import (
    Dataset,
    DatasetType,
    FeatureDatasetDependency,
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

    @patch("ml_platform.data_sdk.materializers._transform_materializer.F")
    def test_reads_each_parent_at_its_pinned_snapshot(
        self, functions: MagicMock
    ) -> None:
        """A newer table head cannot replace the parent snapshot selected by the API."""
        spark = MagicMock()
        frame = spark.read.option.return_value.table.return_value.where.return_value
        frame.dropDuplicates.return_value = frame
        callbacks = [
            MaterializerPartition(
                partition_id="output",
                timestamp=datetime(2026, 10, 19, tzinfo=timezone.utc),
                callback_token="token",
                partition_dependencies=[
                    PartitionDependency(
                        partition_id="parent",
                        dataset_name="sp500",
                        dataset_version=3,
                        timestamp=datetime(2026, 10, 19, tzinfo=timezone.utc),
                        snapshot_id=42,
                    ),
                    PartitionDependency(
                        partition_id="other-parent",
                        dataset_name="company_facts",
                        dataset_version=2,
                        timestamp=datetime(2026, 10, 19, tzinfo=timezone.utc),
                        snapshot_id=99,
                    ),
                ],
            )
        ]

        result = FeatureMaterializer._planned_dataset(
            spark,
            callbacks,
            FeatureDatasetDependency(
                name="sp500",
                version=3,
                dataset_type=DatasetType.MARKET_DATA,
                argument_name="market_data",
                num_lookback_partitions=0,
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
                FeatureDatasetDependency(
                    name="sp500",
                    version=3,
                    dataset_type=DatasetType.MARKET_DATA,
                    argument_name="market_data",
                    num_lookback_partitions=0,
                ),
            )

    @patch.object(FeatureMaterializer, "_planned_dataset")
    def test_injects_each_declared_dataset_under_its_argument_name(
        self, planned_dataset: MagicMock
    ) -> None:
        """Multiple datasets retain their own identity, snapshot plan, and constructor slot."""
        timestamp = datetime(2026, 10, 19, tzinfo=timezone.utc)
        config = FeatureMaterializationConfig.model_validate(
            {
                "name": "joined_feature",
                "version": 1,
                "description": "Joined feature.",
                "warehouse_dir": "temporary-warehouse",
                "dtype": "double",
                "frequency": "daily",
                "transform": "ml_platform.data_sdk.feature_transforms._joined.JoinedTransform",
                "datasets": [
                    {
                        "name": "prices_v2",
                        "version": 1,
                        "dataset_type": DatasetType.MARKET_DATA,
                        "argument_name": "prices",
                        "num_lookback_partitions": 1,
                    },
                    {
                        "name": "company_facts",
                        "version": 2,
                        "dataset_type": DatasetType.MARKET_DATA,
                        "argument_name": "fundamentals",
                        "num_lookback_partitions": 0,
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
                                "dataset_name": "prices_v2",
                                "dataset_version": 1,
                                "timestamp": timestamp,
                                "snapshot_id": 42,
                            },
                            {
                                "partition_id": "facts-parent",
                                "dataset_name": "company_facts",
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
        self.assertEqual(planned_dataset.call_args_list[0].args[2].name, "prices_v2")
        self.assertEqual(
            planned_dataset.call_args_list[1].args[2].name, "company_facts"
        )


class TransformMaterializerTest(unittest.TestCase):
    """Verify feature and label workers use the same pinned-dataset mixin."""

    @patch.object(LabelMaterializer, "_planned_dataset")
    @patch.object(
        LabelMaterializer, "_load_transform", return_value=TripleBarrierLabelTransform
    )
    def test_label_injects_pinned_datasets_through_the_shared_mixin(
        self,
        load_transform: MagicMock,
        planned_dataset: MagicMock,
    ) -> None:
        """Labels use the same dataset read and constructor injection contract as features."""
        timestamp = datetime(2026, 10, 19, tzinfo=timezone.utc)
        config = LabelMaterializationConfig.model_validate(
            {
                "name": "triple_barrier",
                "version": 1,
                "description": "Labels.",
                "warehouse_dir": "temporary-warehouse",
                "dtype": "int",
                "frequency": "daily",
                "feature_keys": ["symbol"],
                "datasets": [
                    {
                        "name": "sp500",
                        "version": 3,
                        "dataset_type": DatasetType.MARKET_DATA,
                        "argument_name": "market_data",
                        "num_lookback_partitions": 1,
                    }
                ],
                "transform": "ml_platform.data_sdk.label_transforms._triple_barrier.TripleBarrierLabelTransform",
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
                                "dataset_name": "sp500",
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
