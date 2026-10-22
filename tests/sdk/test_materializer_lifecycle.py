"""Spark/Iceberg integration coverage for the shared materializer lifecycle."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from ml_platform.data_sdk.materializers import (
    DatasetMaterializationConfig,
    DatasetMaterializer,
    FeatureMaterializationConfig,
    FeatureMaterializer,
    LabelMaterializationConfig,
    LabelMaterializer,
)
from ml_platform.data_sdk.transforms._types import Dataset, DatasetReference


class MaterializerLifecycleIntegrationTest(unittest.TestCase):
    """Verify runtime workers write and report actual Iceberg revisions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = TemporaryDirectory()
        cls.warehouse = Path(cls.temporary_directory.name) / "warehouse"
        cls.spark = DatasetMaterializer._create_spark_session(cls.warehouse)
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()
        cls.temporary_directory.cleanup()

    def _assert_resource_properties(self, worker: object, resource_type: str, frequency: str) -> None:
        """Read back shared metadata from the committed Iceberg table."""
        metadata_directory = (
            self.warehouse / worker.iceberg_namespace() / worker.config._iceberg_table_name / "metadata"
        )
        metadata_file = max(
            metadata_directory.glob("v*.metadata.json"),
            key=lambda path: int(path.name.split(".", maxsplit=1)[0][1:]),
        )
        properties = json.loads(metadata_file.read_text())["properties"]
        self.assertEqual(properties["resource.type"], resource_type)
        self.assertEqual(properties["resource.name"], worker.config.name)
        self.assertEqual(properties["resource.version"], str(worker.config.version))
        self.assertEqual(properties["resource.frequency"], frequency)
        self.assertEqual(properties["resource.description"], worker.config.description)
        self.assertEqual(
            json.loads(properties["resource.dependencies"]),
            [dataset.model_dump(mode="json") for dataset in worker.config.datasets],
        )
        self.assertEqual(
            json.loads(properties["resource.validations"]),
            worker.config.validations.model_dump(mode="json"),
        )

    def test_dataset_worker_writes_rows_and_reports_empty_requested_day(self) -> None:
        """The generic lifecycle sends one batch for all requested days after a real write."""
        config = DatasetMaterializationConfig.model_validate_json(json.dumps({
            "name": "lifecycle_market", "version": 1,
            "description": "Synthetic market bars.", "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily",
            "symbols": ["AAA"], "warehouse_dir": str(self.warehouse),
            "partitions": [
                {"partition_id": "first", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "one"},
                {"partition_id": "second", "timestamp": "2026-01-02T00:00:00Z", "callback_token": "two"},
            ],
        }))
        source = self.spark.createDataFrame(
            [("2026-01-01 15:30:00", "AAA", "regular", 100.0, 101.0, 99.0, 100.5, 1000)],
            "timestamp string, symbol string, market_session string, Open double, High double, "
            "Low double, Close double, Volume long",
        )
        worker = DatasetMaterializer(config)
        transform = MagicMock(
            iceberg_schema=(
                "timestamp TIMESTAMP, symbol STRING, market_session STRING, Open DOUBLE, "
                "High DOUBLE, Low DOUBLE, Close DOUBLE, Volume BIGINT"
            ),
            iceberg_partition_spec="days(timestamp)",
        )
        transform.return_value = source
        with (
            patch.object(type(worker), "_create_spark_session", return_value=self.spark),
            patch.object(type(worker), "_stop_spark_session"),
            patch.object(type(worker), "_create_transform", return_value=transform),
            patch("ml_platform.data_sdk.materializers._base.materialization_api_url", return_value="http://api"),
            patch("ml_platform.data_sdk.materializers._base.urlopen") as open_url,
        ):
            self.assertIsNone(worker())

        self.assertEqual(self.spark.table(worker._table_identifier).count(), 1)
        self._assert_resource_properties(worker, "dataset", "daily")
        worker._ensure_table(
            self.spark,
            schema=transform.iceberg_schema,
            partition_spec=transform.iceberg_partition_spec,
        )
        self._assert_resource_properties(worker, "dataset", "daily")
        open_url.assert_called_once()
        receipts = json.loads(open_url.call_args.args[0].data)
        self.assertEqual(len(receipts), 2)
        self.assertEqual(receipts[1]["content_hash"], sha256(b"").hexdigest())
        self.assertEqual([receipt["row_count"] for receipt in receipts], [1, 0])
        self.assertTrue(all(receipt["snapshot_id"] > 0 for receipt in receipts))

    def test_feature_worker_writes_requested_hour_and_reports_parent(self) -> None:
        """Feature validation, partition overwrite, and lineage callback run in order."""
        config = FeatureMaterializationConfig.model_validate_json(json.dumps({
            "name": "lifecycle_feature", "version": 1,
            "description": "Synthetic hourly feature.", "dtype": "double",
            "frequency": "hourly", "feature_keys": ["symbol"],
            "datasets": [{"name": "lifecycle_market", "version": 1,
                          "argument_name": "market_data", "num_lookback_partitions": 0}],
            "transform": "ml_platform.data_sdk.feature_transforms._relative_volume.RelativeVolumeTransform",
            "parameters": {"lookback_bars": 2},
            "warehouse_dir": str(self.warehouse),
            "partitions": [{
                "partition_id": "feature-hour", "timestamp": "2026-01-01T10:00:00Z",
                "callback_token": "feature-token",
                "partition_dependencies": [{
                    "partition_id": "market-day", "dataset_name": "lifecycle_market",
                    "dataset_version": 1, "timestamp": "2026-01-01T00:00:00Z", "snapshot_id": 42,
                }],
            }],
        }))
        input_keys = self.spark.createDataFrame([("AAA",)], "symbol string")
        source = self.spark.createDataFrame(
            [("2026-01-01 10:30:00", "AAA", 1.25),
             ("2026-01-01 11:30:00", "AAA", 2.50)],
            "timestamp string, symbol string, feature_value double",
        )
        worker = FeatureMaterializer(config)
        transform = MagicMock(
            iceberg_schema="timestamp TIMESTAMP, symbol STRING, feature_value DOUBLE",
            iceberg_partition_spec="hours(timestamp)",
        )
        transform.return_value = source
        datasets = {"market_data": Dataset(DatasetReference("lifecycle_market", 1, 0), input_keys)}
        with (
            patch.object(type(worker), "_create_spark_session", return_value=self.spark),
            patch.object(type(worker), "_stop_spark_session"),
            patch.object(type(worker), "_create_transform", return_value=transform),
            patch.object(type(worker), "_load_planned_datasets", return_value=datasets),
            patch("ml_platform.data_sdk.materializers._base.materialization_api_url", return_value="http://api"),
            patch("ml_platform.data_sdk.materializers._base.urlopen") as open_url,
        ):
            self.assertIsNone(worker())

        rows = self.spark.table(worker._table_identifier).collect()
        self._assert_resource_properties(worker, "feature", "hourly")
        self.assertNotIn(
            "feature.name",
            dict(self.spark.sql(f"SHOW TBLPROPERTIES {worker._table_identifier}").collect()),
        )
        self.assertEqual([(row.symbol, row.feature_value) for row in rows], [("AAA", 1.25)])
        request = open_url.call_args.args[0]
        self.assertEqual(json.loads(request.data)[0]["dataset_partition_ids"], ["market-day"])
        self.assertGreater(json.loads(request.data)[0]["snapshot_id"], 0)

    def test_label_worker_uses_shared_table_and_merge_lifecycle(self) -> None:
        """Label persistence and callback use the same base lifecycle."""
        config = LabelMaterializationConfig.model_validate_json(json.dumps({
            "name": "lifecycle_label", "version": 1,
            "description": "Synthetic label.", "warehouse_dir": str(self.warehouse),
            "datasets": [{"name": "lifecycle_market", "version": 1, "num_lookback_partitions": 0}],
            "transform": "ml_platform.data_sdk.label_transforms._triple_barrier.TripleBarrierLabelTransform",
            "partitions": [{
                "partition_id": "label-day", "timestamp": "2026-01-01T00:00:00Z",
                "callback_token": "label-token",
                "partition_dependencies": [{"partition_id": "market-day", "dataset_name": "lifecycle_market", "dataset_version": 1, "timestamp": "2026-01-01T00:00:00Z", "snapshot_id": 42}],
            }],
        }))
        source = self.spark.createDataFrame(
            [("2026-01-01 10:30:00", "AAA", 1)],
            "timestamp string, symbol string, label_value int",
        ).selectExpr("cast(timestamp as timestamp) as timestamp", "symbol", "label_value")
        worker = LabelMaterializer(config, datasets={})
        transform = MagicMock()
        transform.return_value = source
        with (
            patch.object(type(worker), "_create_spark_session", return_value=self.spark),
            patch.object(type(worker), "_stop_spark_session"),
            patch.object(type(worker), "_create_transform", return_value=transform),
            patch("ml_platform.data_sdk.materializers._base.materialization_api_url", return_value="http://api"),
            patch("ml_platform.data_sdk.materializers._base.urlopen") as open_url,
        ):
            self.assertIsNone(worker())

        rows = self.spark.table(worker._table_identifier).collect()
        self._assert_resource_properties(worker, "label", "daily")
        self.assertEqual([(row.symbol, row.label_value) for row in rows], [("AAA", 1)])
        self.assertEqual(json.loads(open_url.call_args.args[0].data)[0]["dataset_partition_ids"], ["market-day"])


if __name__ == "__main__":
    unittest.main()
