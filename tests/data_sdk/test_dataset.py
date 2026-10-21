"""Unit tests for typed dataset worker dispatch."""

import unittest
from unittest.mock import patch

from ml_platform.data_sdk.dataset import materialize_dataset
from ml_platform.sdk.configs import DatasetType


class DatasetWorkerTest(unittest.TestCase):
    """Verify registered dataset types dispatch to their Spark implementations."""

    @patch("ml_platform.data_sdk.dataset.materialize_market_data_with_receipts")
    def test_market_data_dispatches_to_spark_materializer(self, materialize: object) -> None:
        """Pass validated market-data config and the target warehouse to the worker."""
        materialize.return_value.row_count = 12
        result = materialize_dataset(
            DatasetType.MARKET_DATA,
            {
                "symbols": ["AAPL"],
                "time_range": {"start_date": "2026-01-01", "end_date": "2026-01-02"},
            },
            warehouse_dir="temporary-warehouse",
        )

        self.assertEqual(result, 12)
        materialize.assert_called_once()
        config = materialize.call_args.args[0]
        self.assertEqual(config.dataset_type, DatasetType.MARKET_DATA)
        self.assertEqual(config.symbols, ["AAPL"])
        self.assertEqual(str(materialize.call_args.kwargs["warehouse_dir"]), "temporary-warehouse")


if __name__ == "__main__":
    unittest.main()
