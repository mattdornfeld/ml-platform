"""Unit tests for API-dispatched dataset materialization."""

import json
import unittest
from unittest.mock import MagicMock, patch

from ml_platform.common.types import Environment
from ml_platform.data_sdk.materializers import (
    DatasetMaterializationConfig,
    DatasetMaterializer,
    IcebergTable,
)


class DatasetWorkerTest(unittest.TestCase):
    """Verify the dataset worker dispatches a transform and declares its table."""

    def config(
        self, dataset_type: str = "market_data"
    ) -> DatasetMaterializationConfig:
        """Build one strict dataset job configuration."""
        return DatasetMaterializationConfig.model_validate_json(
            json.dumps(
                {
                    "environment": "local",
                    "name": "sp500",
                    "version": 3,
                    "description": "Test market data.",
                    "dataset_type": dataset_type,
                    "keys": ["symbol"],
                    "partition_type": "daily",
                    "symbols": ["AAPL"],
                    "validations": [],
                    "warehouse_dir": "temporary-warehouse",
                    "partitions": [
                        {
                            "partition_id": "partition-1",
                            "timestamp": "2026-01-01T00:00:00Z",
                            "callback_token": "token",
                        }
                    ],
                }
            )
        )

    @patch("ml_platform.data_sdk.materializers._dataset.MarketDataTransform")
    def test_market_data_dispatches_to_transform(
        self, transform: MagicMock
    ) -> None:
        """The declared dataset type constructs MarketDataTransform with job partitions."""
        worker = DatasetMaterializer(self.config())
        worker._transform(MagicMock())
        self.assertEqual(transform.call_args.kwargs["symbols"], ("AAPL",))
        self.assertEqual(
            transform.call_args.kwargs["partitions"][0].date().isoformat(),
            "2026-01-01",
        )

    @patch("ml_platform.data_sdk.materializers._dataset.MarketDataTransform")
    def test_test_environment_uses_the_synthetic_dataset_provider(
        self, transform: MagicMock
    ) -> None:
        """The materializer selects deterministic data only through Environment.TEST."""
        config = self.config().model_copy(
            update={"environment": Environment.TEST}
        )

        dataset_transform = DatasetMaterializer(config)._transform(MagicMock())

        self.assertEqual(
            type(dataset_transform).__name__, "SyntheticDatasetTransform"
        )
        self.assertIs(
            dataset_transform.source_transform, transform.return_value
        )

    def test_unsupported_dataset_type_is_rejected_by_configuration(
        self,
    ) -> None:
        """Unknown dataset types cannot enter the materializer contract."""
        with self.assertRaises(ValueError):
            self.config("unsupported")

    def test_dataset_declares_iceberg_table(self) -> None:
        """Dataset persistence owns an immutable Iceberg table definition."""
        table = DatasetMaterializer(self.config()).iceberg_table
        self.assertIsInstance(table, IcebergTable)
        self.assertEqual(table.partition_type.value, "daily")


if __name__ == "__main__":
    unittest.main()
