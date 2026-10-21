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

    @patch.object(DatasetMaterializer, "_materialize_frame", return_value=12)
    @patch("ml_platform.data_sdk.dataset_transforms._market_data.MarketDataTransform")
    @patch.object(DatasetMaterializer, "_create_spark_session")
    def test_market_data_dispatches_to_transform(self, session: MagicMock, transform: MagicMock, materialize: MagicMock) -> None:
        """Deserialize the API JSON contract and pass it to the transform constructor."""
        spark = session.return_value
        spark.sql.return_value.first.return_value = {"snapshot_id": 7}
        transform.return_value.return_value = MagicMock()
        config = DatasetMaterializer._config_from_json(json.dumps({
            "name": "sp500", "version": 3, "dataset_type": "market_data",
            "description": "Test market data.", "validations": {}, "symbols": ["AAPL"],
            "warehouse_dir": "temporary-warehouse",
            "partitions": [{"partition_id": "partition-1", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token"}],
        }))
        self.assertIsInstance(config, DatasetMaterializationConfig)
        with patch("ml_platform.data_sdk.materializers._dataset.F"), patch.object(
            DatasetMaterializer, "_complete_partitions"
        ):
            result = DatasetMaterializer(config)()
        self.assertEqual(result, 12)
        self.assertEqual(transform.call_args.kwargs["symbols"], ["AAPL"])
        self.assertEqual(transform.call_args.kwargs["partitions"][0].date().isoformat(), "2026-01-01")
        materialize.assert_called_once()
        spark.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
