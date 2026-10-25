"""Materializer lifecycle tests using immutable Iceberg table definitions."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from ml_platform.common.types import PartitionType
from ml_platform.data_sdk.materializers import (
    DatasetMaterializationConfig,
    DatasetMaterializer,
    IcebergTable,
)


def dataset_config() -> DatasetMaterializationConfig:
    """Build one strict API-delivered dataset materialization config."""
    return DatasetMaterializationConfig.model_validate_json(
        json.dumps(
            {
                "name": "sp500",
                "version": 3,
                "description": "Hourly market bars.",
                "environment": "local",
                "dataset_type": "market_data",
                "keys": ["symbol"],
                "partition_type": "daily",
                "symbols": ["AAPL"],
                "validations": [],
                "warehouse_dir": "temporary-warehouse",
                "partitions": [
                    {
                        "partition_id": "dataset-partition",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "callback_token": "dataset-token",
                    }
                ],
            }
        )
    )


class IcebergTableTest(unittest.TestCase):
    """Verify table identity, metadata, and write behavior live on IcebergTable."""

    def table(self) -> IcebergTable:
        """Return a concise immutable table declaration for this test."""
        return IcebergTable(
            catalog="ml_platform",
            namespace="features",
            name="returns",
            version=1,
            resource_type="feature",
            schema={
                "timestamp": "timestamp",
                "symbol": "string",
                "feature_value": "double",
            },
            partition_type=PartitionType.HOURLY,
            description="Feature returns.",
            dependencies=(),
            validations=(),
            additional_properties=(),
        )

    def test_id_includes_catalog_namespace_and_version(self) -> None:
        """A table owns its fully-qualified immutable identifier."""
        self.assertEqual(self.table().id, "ml_platform.features.`returns-v1`")

    def test_storage_path_uses_the_table_identity(self) -> None:
        """A table owns its physical warehouse location."""
        self.assertEqual(
            self.table().storage_path("temporary-warehouse"),
            Path("temporary-warehouse").resolve() / "features" / "returns-v1",
        )

    def test_create_applies_configured_properties(self) -> None:
        """Table creation uses only the table's injected schema and metadata."""
        table = self.table()
        spark = MagicMock()
        spark.catalog.tableExists.return_value = False
        writer = MagicMock()
        writer.tableProperty.side_effect = lambda *_: writer
        writer.partitionedBy.return_value = writer
        spark.createDataFrame.return_value.writeTo.return_value.using.return_value = writer
        with patch.object(
            IcebergTable, "_partition_transform", return_value=MagicMock()
        ):
            table.create(spark)
        spark.createDataFrame.assert_called_once_with(
            [], "timestamp TIMESTAMP, symbol STRING, feature_value DOUBLE"
        )
        writer.tableProperty.assert_any_call("resource.type", "feature")
        writer.create.assert_called_once_with()

    def test_overwrite_writes_only_emitted_partitions(self) -> None:
        """An overwrite table owns dynamic partition replacement."""
        data_frame = MagicMock()
        self.table().write(data_frame)
        data_frame.writeTo.assert_called_once_with(
            "ml_platform.features.`returns-v1`"
        )
        data_frame.writeTo.return_value.overwritePartitions.assert_called_once_with()


class MaterializerLifecycleTest(unittest.TestCase):
    """Verify the base worker delegates persistence to its declared Iceberg table."""

    def test_main_validates_then_creates_and_writes_declared_table(
        self,
    ) -> None:
        """Worker lifecycle owns transforms and callbacks while the table owns storage."""
        config = dataset_config()
        spark = MagicMock()
        transform = MagicMock()
        data_frame = MagicMock()
        table = MagicMock()
        table.snapshot_id.return_value = 42
        with (
            patch.object(
                DatasetMaterializer, "_create_spark_session", return_value=spark
            ),
            patch.object(
                DatasetMaterializer, "_transform", return_value=transform
            ),
            patch.object(
                DatasetMaterializer,
                "_requested_partitions",
                return_value=MagicMock(),
            ),
            patch.object(
                DatasetMaterializer,
                "_filter_partitions",
                return_value=data_frame,
            ),
            patch.object(DatasetMaterializer, "validate") as validate,
            patch.object(
                DatasetMaterializer, "_complete_partitions"
            ) as complete,
            patch.object(
                DatasetMaterializer,
                "iceberg_table",
                new_callable=PropertyMock,
                return_value=table,
            ),
            patch.object(
                sys,
                "argv",
                ["dataset", "--config-json", config.model_dump_json()],
            ),
            patch("builtins.print"),
        ):
            DatasetMaterializer.main()
        transform.assert_called_once_with()
        validate.assert_called_once_with(data_frame)
        table.create.assert_called_once_with(spark)
        table.write.assert_called_once_with(data_frame)
        complete.assert_called_once_with(spark, 42)
        spark.stop.assert_called_once_with()

    @patch("ml_platform.data_sdk.materializers._base.httpx2.request")
    def test_partition_completion_posts_all_partition_receipts(
        self, request: MagicMock
    ) -> None:
        """Workers complete Iceberg partitions through the shared callback endpoint."""
        spark = MagicMock()
        table = MagicMock()
        table.resource_type = "dataset"
        table.partition_keys.return_value = ["2026-01-01"]
        table.partition_content_hashes.return_value = {"2026-01-01": "digest"}
        table.partition_row_counts.return_value = {"2026-01-01": 5}
        table.schema_fields.return_value = {"timestamp": "timestamp"}
        table.empty_content_hash.return_value = "empty"
        materializer = DatasetMaterializer(dataset_config())
        with (
            patch.object(
                DatasetMaterializer,
                "iceberg_table",
                new_callable=PropertyMock,
                return_value=table,
            ),
            patch.object(
                DatasetMaterializer, "_row_digest", return_value=MagicMock()
            ),
        ):
            materializer._complete_partitions(spark, 42)

        response = request.return_value
        response.raise_for_status.assert_called_once_with()
        receipt = json.loads(request.call_args.kwargs["content"])[0]
        self.assertEqual(receipt["partition_id"], "dataset-partition")
        self.assertEqual(receipt["snapshot_id"], 42)
        self.assertEqual(receipt["row_count"], 5)

    def test_dataset_materializer_declares_market_data_table(self) -> None:
        """Dataset-specific metadata is injected into the table declaration."""
        table = DatasetMaterializer(dataset_config()).iceberg_table
        self.assertEqual(table.namespace, "market_data")
        self.assertEqual(table.partition_type, PartitionType.DAILY)
        self.assertEqual(table.resource_type, "dataset")


if __name__ == "__main__":
    unittest.main()
