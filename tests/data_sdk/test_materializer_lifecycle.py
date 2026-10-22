"""Shared Iceberg write and control-plane callback lifecycle tests."""

from hashlib import sha256
from dataclasses import FrozenInstanceError
import json
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from ml_platform.data_sdk.materializers import (
    DatasetMaterializationConfig,
    DatasetMaterializer,
    FeatureMaterializationConfig,
    FeatureMaterializer,
    LabelMaterializationConfig,
    LabelMaterializer,
)
from ml_platform.data_sdk.materializers._base import MaterializerPartition


def dataset_worker() -> DatasetMaterializer:
    """Build a runtime dataset worker with one requested partition."""
    config = DatasetMaterializationConfig.model_validate_json(json.dumps({
        "name": "sp500",
        "version": 3,
        "description": "Hourly market bars.",
        "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily",
        "symbols": ["AAPL"],
        "validations": [],
        "warehouse_dir": "temporary-warehouse",
        "partitions": [{
            "partition_id": "dataset-partition",
            "timestamp": "2026-01-01T00:00:00Z",
            "callback_token": "dataset-token",
        }],
    }))
    return DatasetMaterializer(config)


def feature_worker() -> FeatureMaterializer:
    """Build a runtime feature worker with one pinned parent partition."""
    config = FeatureMaterializationConfig.model_validate_json(json.dumps({
        "name": "log_return",
        "version": 1,
        "description": "One-bar log returns.",
        "dtype": "double",
        "frequency": "hourly",
        "feature_keys": ["symbol"],
        "datasets": [{
            "name": "sp500", "version": 3,
            "argument_name": "market_data", "num_lookback_partitions": 0,
        }],
        "transform": "ml_platform.data_sdk.feature_transforms._relative_volume.RelativeVolumeTransform",
        "parameters": {"lookback_bars": 2},
        "validations": [{
            "type": "minimum_rows",
            "minimum_rows": 1,
            "reason": "Each requested feature partition requires a row.",
        }],
        "warehouse_dir": "temporary-warehouse",
        "partitions": [{
            "partition_id": "feature-partition",
            "timestamp": "2026-01-01T10:00:00Z",
            "callback_token": "feature-token",
            "partition_dependencies": [{
                "partition_id": "parent-partition",
                "dataset_name": "sp500",
                "dataset_version": 3,
                "timestamp": "2026-01-01T00:00:00Z",
                "snapshot_id": 42,
            }],
        }],
    }))
    return FeatureMaterializer(config)


def label_worker() -> LabelMaterializer:
    """Build a label worker with one declared and pinned market dependency."""
    config = LabelMaterializationConfig.model_validate_json(json.dumps({
        "name": "triple_barrier", "version": 1,
        "description": "Daily triple-barrier labels.",
        "warehouse_dir": "temporary-warehouse",
        "datasets": [{"name": "sp500", "version": 3, "num_lookback_partitions": 1}],
        "transform": "ml_platform.data_sdk.label_transforms._triple_barrier.TripleBarrierLabelTransform",
        "partitions": [{
            "partition_id": "label-partition", "timestamp": "2026-01-01T00:00:00Z",
            "callback_token": "label-token",
            "partition_dependencies": [{
                "partition_id": "parent-partition", "dataset_name": "sp500",
                "dataset_version": 3, "timestamp": "2026-01-01T00:00:00Z",
                "snapshot_id": 42,
            }],
        }],
    }))
    return LabelMaterializer(config, datasets=())


def run_materializer_main(worker: DatasetMaterializer) -> None:
    """Run the standard worker CLI with this test worker's serialized config."""
    with patch.object(sys, "argv", ["materializer", "--config-json", worker.config.model_dump_json()]):
        type(worker).main()


class MaterializerLifecycleTest(unittest.TestCase):
    """Verify the generic base owns table setup, persistence, and callbacks."""

    def test_base_validates_before_writing_and_reports_after_commit(self) -> None:
        """The shared lifecycle orders validation, Iceberg writes, and callbacks."""
        worker = dataset_worker()
        spark = MagicMock()
        source = MagicMock()
        prepared = MagicMock()
        transform = MagicMock(iceberg_schema="timestamp TIMESTAMP", iceberg_partition_spec="days(timestamp)")
        calls = MagicMock()
        with (
            patch.object(type(worker), "_create_spark_session", return_value=spark),
            patch.object(type(worker), "_create_transform", return_value=transform),
            patch.object(type(worker), "materialize", return_value=source) as materialize,
            patch.object(type(worker), "_prepare_frame", return_value=prepared) as prepare,
            patch.object(type(worker), "validate") as validate,
            patch.object(type(worker), "_create_table") as ensure,
            patch.object(type(worker), "_write_to_table") as write,
            patch.object(type(worker), "_latest_snapshot_id", return_value=42) as latest_snapshot,
            patch.object(type(worker), "_complete_partitions") as complete,
        ):
            for name, method in (
                ("materialize", materialize), ("prepare", prepare), ("validate", validate),
                ("ensure", ensure), ("write", write), ("snapshot", latest_snapshot),
                ("complete", complete),
            ):
                calls.attach_mock(method, name)
            run_materializer_main(worker)

        self.assertEqual(calls.mock_calls, [
            call.materialize(spark, transform),
            call.prepare(source),
            call.validate(prepared),
            call.ensure(spark, schema="timestamp TIMESTAMP", partition_spec="days(timestamp)"),
            call.write(prepared),
            call.snapshot(spark, "ml_platform.market_data.`sp500-v3`"),
            call.complete(spark, 42, "days(timestamp)"),
        ])
        spark.stop.assert_called_once_with()

    def test_validation_failure_does_not_write_or_report(self) -> None:
        """A malformed transform result cannot produce an Iceberg revision or receipt."""
        worker = dataset_worker()
        spark = MagicMock()
        with (
            patch.object(type(worker), "_create_spark_session", return_value=spark),
            patch.object(type(worker), "materialize", return_value=MagicMock()),
            patch.object(type(worker), "_prepare_frame", return_value=MagicMock()),
            patch.object(type(worker), "validate", side_effect=ValueError("invalid")),
            patch.object(type(worker), "_write_to_table") as write,
            patch.object(type(worker), "_complete_partitions") as complete,
        ):
            with self.assertRaisesRegex(ValueError, "invalid"):
                run_materializer_main(worker)

        write.assert_not_called()
        complete.assert_not_called()
        spark.stop.assert_called_once_with()

    def test_feature_table_uses_definition_metadata(self) -> None:
        """The table creator persists feature-specific schema and properties."""
        worker = feature_worker()
        spark = MagicMock()
        spark.catalog.tableExists.return_value = False
        table_writer = MagicMock()
        table_writer.tableProperty.side_effect = lambda *_: table_writer
        table_writer.partitionedBy.return_value = table_writer
        spark.createDataFrame.return_value.writeTo.return_value.using.return_value = table_writer

        with patch.object(type(worker), "_partition_transform", return_value=MagicMock()):
            worker._create_table(
                spark,
                schema="timestamp TIMESTAMP, symbol STRING, feature_value DOUBLE",
                partition_spec="hours(timestamp)",
            )

        spark.createDataFrame.assert_called_once_with(
            [], "timestamp TIMESTAMP, symbol STRING, feature_value DOUBLE"
        )
        table_properties = {
            call.args[0]: call.args[1]
            for call in table_writer.tableProperty.call_args_list
        }
        self.assertEqual(table_properties["feature.transform"], worker.config.transform)
        table_writer.partitionedBy.assert_called_once()
        table_writer.create.assert_called_once_with()

    def test_every_resource_uses_shared_table_properties(self) -> None:
        """Resource identity, cadence, lineage and validations come from the base."""
        for resource_type, worker, frequency in (
            ("dataset", dataset_worker(), "daily"),
            ("feature", feature_worker(), "hourly"),
            ("label", label_worker(), "daily"),
        ):
            with self.subTest(resource_type=resource_type):
                properties = worker._table_properties("hours(timestamp)" if resource_type == "feature" else "days(timestamp)")
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
                    [validation.model_dump(mode="json") for validation in worker.config.validations],
                )
                self.assertNotIn(f"{resource_type}.name", properties)

    def test_existing_table_properties_are_reconciled(self) -> None:
        """A worker adds missing metadata to a previously created Iceberg table."""
        worker = dataset_worker()
        spark = MagicMock()
        spark.catalog.tableExists.return_value = True
        spark.sql.return_value.collect.return_value = [
            SimpleNamespace(key="resource.name", value="sp500"),
        ]

        worker._create_table(spark, schema="timestamp TIMESTAMP", partition_spec="days(timestamp)")

        commands = [call.args[0] for call in spark.sql.call_args_list]
        self.assertIn("SHOW TBLPROPERTIES", commands[0])
        self.assertIn("ALTER TABLE", commands[1])
        self.assertIn("'resource.type' = 'dataset'", commands[1])
        self.assertIn("'resource.name' = 'sp500'", commands[1])

    def test_existing_resource_metadata_does_not_create_another_revision(self) -> None:
        """A repeated job does not rewrite immutable table properties."""
        worker = feature_worker()
        spark = MagicMock()
        spark.catalog.tableExists.return_value = True
        spark.sql.return_value.collect.return_value = [
            SimpleNamespace(key="resource.type", value="feature"),
            SimpleNamespace(key="resource.description", value="*********(redacted)"),
            SimpleNamespace(key="resource.frequency", value="*********(redacted)"),
        ]

        worker._create_table(spark, schema="timestamp TIMESTAMP", partition_spec="hours(timestamp)")

        spark.sql.assert_called_once_with(f"SHOW TBLPROPERTIES {worker._table_identifier}")

    def test_materialize_only_constructs_requested_feature_frame(self) -> None:
        """Feature construction leaves persistence to the base lifecycle."""
        worker = feature_worker()
        datasets = {"market_data": MagicMock()}
        requested = MagicMock()
        feature = MagicMock()
        filtered_feature = MagicMock()
        with (
            patch.object(type(worker), "_requested_partitions", return_value=requested),
            patch.object(type(worker), "_filter_partitions", return_value=filtered_feature) as filter_partitions,
            patch.object(type(worker), "_write_to_table") as write,
        ):
            transform = MagicMock(iceberg_partition_spec="hours(timestamp)")
            transform.return_value = feature
            result = worker.materialize(MagicMock(), transform)

        self.assertIs(result, filtered_feature)
        self.assertFalse(hasattr(worker, "_input_datasets"))
        filter_partitions.assert_called_once_with(feature, requested, "hours(timestamp)")
        write.assert_not_called()

    def test_materializer_instance_is_frozen(self) -> None:
        """Runtime workers cannot acquire mutable instance caches."""
        worker = feature_worker()
        with self.assertRaises(FrozenInstanceError):
            worker._input_datasets = {}  # type: ignore[attr-defined]

    def test_feature_write_replaces_only_emitted_iceberg_partitions(self) -> None:
        """The shared write strategy uses Iceberg dynamic partition replacement."""
        worker = feature_worker()
        frame = MagicMock()
        worker._write_to_table(frame)
        frame.writeTo.assert_called_once_with("ml_platform.features.`log_return-v1`")
        frame.writeTo.return_value.overwritePartitions.assert_called_once_with()

    @patch("ml_platform.data_sdk.materializers._base.urlopen")
    @patch("ml_platform.data_sdk.materializers._base.materialization_api_url", return_value="http://api")
    def test_feature_callback_carries_pinned_parent_ids(
        self, _api_url: MagicMock, open_url: MagicMock
    ) -> None:
        """The generic callback includes lineage for derived feature partitions."""
        worker = feature_worker()
        spark = MagicMock()
        spark.table.return_value.schema.fields = [
            SimpleNamespace(name="timestamp", dataType=SimpleNamespace(simpleString=lambda: "timestamp")),
            SimpleNamespace(name="symbol", dataType=SimpleNamespace(simpleString=lambda: "string")),
            SimpleNamespace(name="feature_value", dataType=SimpleNamespace(simpleString=lambda: "double")),
        ]
        with patch.object(type(worker), "_partition_content_hashes", return_value={"2026-01-01T10:00:00": "feature-hash"}), patch.object(
            type(worker), "_partition_row_counts", return_value={"2026-01-01T10:00:00": 7}
        ):
            worker._complete_partitions(spark, 91, "hours(timestamp)")

        request = open_url.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://api/partitions/complete",
        )
        self.assertEqual(json.loads(request.data), [{
            "partition_id": "feature-partition",
            "resource_type": "feature",
            "row_count": 7,
            "schema": {"timestamp": "timestamp", "symbol": "string", "feature_value": "double"},
            "content_hash": "feature-hash",
            "snapshot_id": 91,
            "callback_token": "feature-token",
            "dataset_partition_ids": ["parent-partition"],
        }])

    @patch("ml_platform.data_sdk.materializers._base.urlopen")
    @patch("ml_platform.data_sdk.materializers._base.materialization_api_url", return_value="http://api")
    def test_empty_dataset_partition_gets_empty_content_hash(
        self, _api_url: MagicMock, open_url: MagicMock
    ) -> None:
        """An empty market day can still receive a committed completion receipt."""
        worker = dataset_worker()
        spark = MagicMock()
        spark.table.return_value.schema.fields = [
            SimpleNamespace(name="timestamp", dataType=SimpleNamespace(simpleString=lambda: "timestamp"))
        ]
        with patch.object(type(worker), "_partition_content_hashes", return_value={}), patch.object(
            type(worker), "_partition_row_counts", return_value={}
        ):
            worker._complete_partitions(spark, 92, "days(timestamp)")

        request = open_url.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://api/partitions/complete",
        )
        self.assertEqual(json.loads(request.data), [{
            "partition_id": "dataset-partition", "resource_type": "dataset",
            "row_count": 0, "schema": {"timestamp": "timestamp"},
            "content_hash": sha256(b"").hexdigest(),
            "snapshot_id": 92,
            "callback_token": "dataset-token",
        }])

    @patch("ml_platform.data_sdk.materializers._base.urlopen")
    @patch("ml_platform.data_sdk.materializers._base.materialization_api_url", return_value="http://api")
    def test_multiple_partitions_use_one_callback_request(self, _api_url: MagicMock, open_url: MagicMock) -> None:
        """One Iceberg job reports all of its committed partitions together."""
        worker = dataset_worker()
        second = MaterializerPartition(
            partition_id="dataset-partition-2", timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            callback_token="dataset-token-2",
        )
        worker = DatasetMaterializer(worker.config.model_copy(update={"partitions": [*worker.config.partitions, second]}))
        spark = MagicMock()
        spark.table.return_value.schema.fields = [
            SimpleNamespace(name="timestamp", dataType=SimpleNamespace(simpleString=lambda: "timestamp"))
        ]
        with patch.object(type(worker), "_partition_content_hashes", return_value={
            "2026-01-01T00:00:00": "hash-1", "2026-01-02T00:00:00": "hash-2"
        }), patch.object(type(worker), "_partition_row_counts", return_value={
            "2026-01-01T00:00:00": 2, "2026-01-02T00:00:00": 3
        }):
            worker._complete_partitions(spark, 77, "days(timestamp)")
        open_url.assert_called_once()
        receipts = json.loads(open_url.call_args.args[0].data)
        self.assertEqual([receipt["partition_id"] for receipt in receipts], ["dataset-partition", "dataset-partition-2"])
        self.assertEqual([receipt["row_count"] for receipt in receipts], [2, 3])


if __name__ == "__main__":
    unittest.main()
