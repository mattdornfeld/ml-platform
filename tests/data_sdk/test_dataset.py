"""Unit tests for API-dispatched dataset worker dispatch."""

import json
import unittest
from unittest.mock import MagicMock, patch

import ml_platform.data_sdk.dataset_transforms._market_data

from ml_platform.data_sdk.materializers._dataset import (
    DatasetMaterializationConfig,
    DatasetMaterializer,
)


class DatasetWorkerTest(unittest.TestCase):
    """Verify API-delivered dataset definitions use the matching transform."""

    @patch("ml_platform.data_sdk.dataset_transforms._market_data.MarketDataTransform")
    @patch.object(DatasetMaterializer, "_create_spark_session")
    def test_market_data_dispatches_to_transform(
        self,
        session: MagicMock,
        transform: MagicMock,
    ) -> None:
        """Deserialize the API JSON contract and pass it to the transform constructor."""
        spark = session.return_value
        transform.return_value.return_value = MagicMock()
        transform.return_value.iceberg_schema = "timestamp TIMESTAMP"
        transform.return_value.iceberg_partition_spec = "days(timestamp)"
        config = DatasetMaterializer._config_from_json(json.dumps({
            "name": "sp500", "version": 3, "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily",
            "description": "Test market data.", "validations": {}, "symbols": ["AAPL"],
            "warehouse_dir": "temporary-warehouse",
            "partitions": [{"partition_id": "partition-1", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token"}],
        }))
        self.assertIsInstance(config, DatasetMaterializationConfig)
        worker = DatasetMaterializer(config)
        with (
            patch.object(type(worker), "_prepare_frame", return_value=MagicMock()) as prepare,
            patch.object(type(worker), "_validate_frame") as validate,
            patch.object(type(worker), "_ensure_table") as ensure,
            patch.object(type(worker), "_write_frame") as write,
            patch.object(type(worker), "_latest_snapshot_id", return_value=7),
            patch.object(type(worker), "_complete_partitions") as complete,
        ):
            result = worker()
        self.assertIsNone(result)
        self.assertEqual(transform.call_args.kwargs["symbols"], ["AAPL"])
        self.assertEqual(transform.call_args.kwargs["partitions"][0].date().isoformat(), "2026-01-01")
        prepare.assert_called_once_with(transform.return_value.return_value)
        validate.assert_called_once()
        ensure.assert_called_once_with(
            spark,
            schema=transform.return_value.iceberg_schema,
            partition_spec=transform.return_value.iceberg_partition_spec,
        )
        write.assert_called_once_with(prepare.return_value)
        complete.assert_called_once_with(spark, 7, transform.return_value.iceberg_partition_spec)
        spark.stop.assert_called_once()

    def test_unsupported_dataset_type_raises_not_implemented(self) -> None:
        """Unknown dataset types fail before any Spark write is attempted."""
        config = DatasetMaterializationConfig.model_validate_json(json.dumps({
            "name": "unsupported_dataset",
            "version": 1,
            "description": "Unsupported dataset transform test.",
            "dataset_type": "unsupported", "keys": ["symbol"], "partition_type": "daily",
            "symbols": ["AAPL"],
            "warehouse_dir": "temporary-warehouse",
            "partitions": [{
                "partition_id": "partition-1",
                "timestamp": "2026-01-01T00:00:00Z",
                "callback_token": "token",
            }],
        }))
        worker = DatasetMaterializer(config)
        with patch.object(DatasetMaterializer, "_write_frame") as write:
            with self.assertRaisesRegex(NotImplementedError, "Unsupported dataset type: unsupported"):
                worker._create_transform(MagicMock())
        write.assert_not_called()

    @patch("ml_platform.data_sdk.materializers._dataset.F")
    @patch.object(DatasetMaterializer, "_validate_non_empty")
    @patch.object(DatasetMaterializer, "_validate_non_null_columns")
    @patch.object(DatasetMaterializer, "_ensure_table")
    @patch.object(DatasetMaterializer, "_validate_required_columns")
    def test_dataset_frame_uses_dataframe_merge_api(
        self,
        _validate_required_columns: MagicMock,
        _ensure_table: MagicMock,
        _validate_non_null_columns: MagicMock,
        _validate_non_empty: MagicMock,
        _functions: MagicMock,
    ) -> None:
        """Dataset persistence uses the DataFrame merge chain rather than SQL text."""
        frame = MagicMock()
        incoming = frame.select.return_value
        source = MagicMock()
        (
            incoming.withColumn.return_value.repartition.return_value.sortWithinPartitions.return_value.drop.return_value
        ) = source
        aliased_source = source.alias.return_value
        writer = aliased_source.mergeInto.return_value
        writer.whenMatched.return_value.updateAll.return_value.whenNotMatched.return_value.insertAll.return_value.merge.return_value = None
        incoming.count.return_value = 3

        with patch("ml_platform.data_sdk.materializers._base.F"):
            config = DatasetMaterializationConfig.model_validate_json(json.dumps({
                "name": "sp500", "version": 3, "description": "Synthetic market bars.",
                "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily", "symbols": ["AAA"],
                "warehouse_dir": "temporary-warehouse",
                "partitions": [{"partition_id": "synthetic-partition", "timestamp": "2026-01-01T00:00:00Z",
                                "callback_token": "synthetic-token"}],
            }))
            count = DatasetMaterializer._materialize_frame(
                frame,
                spark=MagicMock(),
                config=config,
            )

        self.assertEqual(count, 3)
        _validate_non_null_columns.assert_called_once_with(
            incoming, ("timestamp", "symbol"), context="Dataset transform"
        )
        aliased_source.mergeInto.assert_called_once_with(
            "ml_platform.market_data.`sp500-v3`",
            unittest.mock.ANY,
        )
        writer.whenMatched.return_value.updateAll.assert_called_once()
        writer.whenMatched.return_value.updateAll.return_value.whenNotMatched.return_value.insertAll.assert_called_once()
        writer.whenMatched.return_value.updateAll.return_value.whenNotMatched.return_value.insertAll.return_value.merge.assert_called_once()


if __name__ == "__main__":
    unittest.main()
