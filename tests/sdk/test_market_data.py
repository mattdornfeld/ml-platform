"""Spark and Iceberg coverage for market-data ingestion."""

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ml_platform.data_sdk.configs import BaseConfig, DataConfig, DatasetConfig, LabelConfig, MarketDataConfig, TripleBarrierConfig
from ml_platform.data_sdk.dataset_transforms._market_data import _contiguous_date_ranges, fetch_hourly_ohlcv
from ml_platform.data_sdk.materializers._dataset import DatasetMaterializer

TEST_START_DATE = date(2026, 1, 1)
TEST_END_DATE = date(2026, 1, 10)


def yahoo_payload() -> dict[str, object]:
    """Return a minimal Yahoo chart payload spanning two UTC dates."""
    return {
        "chart": {
            "result": [{
                "timestamp": [1767358800, 1767367800, 1767623400],
                "indicators": {"quote": [{
                    "open": [100.0, 101.0, 102.0], "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0], "close": [100.5, 101.5, 102.5],
                    "volume": [1000, 2000, 3000],
                }]},
            }],
            "error": None,
        }
    }


class MarketDataIntegrationTest(unittest.TestCase):
    """Verify Spark frames are committed as readable Iceberg Parquet data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start a Spark session backed by a disposable Iceberg warehouse."""
        cls.temporary_directory = TemporaryDirectory()
        cls.warehouse = Path(cls.temporary_directory.name) / "warehouse"
        cls.spark = DatasetMaterializer._create_spark_session(cls.warehouse)
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop Spark and remove the temporary warehouse."""
        cls.spark.stop()
        cls.temporary_directory.cleanup()

    def setUp(self) -> None:
        """Drop tables left by the preceding test."""
        if self.spark.catalog.tableExists(DatasetMaterializer._iceberg_table_identifier('sp500', 3)):
            self.spark.sql(f"DROP TABLE {DatasetMaterializer._iceberg_table_identifier('sp500', 3)} PURGE")

    @patch("ml_platform.data_sdk.dataset_transforms._market_data._request_yahoo_chart")
    def test_yahoo_json_writes_spark_dataframe_without_partition_column(self, request: object) -> None:
        """Convert mocked JSON to Spark and persist only the declared table schema."""
        request.return_value = yahoo_payload()
        frame = fetch_hourly_ohlcv(self.spark, "mock", start_date=TEST_START_DATE, end_date=TEST_END_DATE)
        self.assertNotIn("timestamp_day", frame.columns)
        self.assertEqual(DatasetMaterializer._materialize_test_frame(frame, name="sp500", version=3, spark=self.spark), 2)

        rows = self.spark.table(DatasetMaterializer._iceberg_table_identifier('sp500', 3)).orderBy("timestamp").collect()
        self.assertEqual([row.symbol for row in rows], ["MOCK"] * 3)
        self.assertEqual([row.Close for row in rows], [100.5, 101.5, 102.5])
        self.assertNotIn("timestamp_day", self.spark.table(DatasetMaterializer._iceberg_table_identifier('sp500', 3)).columns)
        request.assert_called_once_with("MOCK", TEST_START_DATE, TEST_END_DATE)


class MarketDataUnitTest(unittest.TestCase):
    """Cover market-data configuration validation."""

    def test_market_data_transform_coalesces_adjacent_partitions(self) -> None:
        """Yahoo requests use one half-open range for each contiguous day sequence."""
        ranges = _contiguous_date_ranges([
            date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 4), date(2026, 1, 5),
        ])
        self.assertEqual(ranges, [
            (date(2026, 1, 1), date(2026, 1, 3)),
            (date(2026, 1, 4), date(2026, 1, 6)),
        ])

    def test_configuration_ontology(self) -> None:
        """Specialized resource configs inherit the strict common data contract."""
        self.assertTrue(issubclass(MarketDataConfig, DatasetConfig))
        self.assertTrue(issubclass(DatasetConfig, DataConfig))
        self.assertTrue(issubclass(TripleBarrierConfig, LabelConfig))
        self.assertTrue(issubclass(LabelConfig, DataConfig))
        self.assertTrue(issubclass(DataConfig, BaseConfig))

    def test_config_rejects_invalid_and_aliased_dates(self) -> None:
        """Require the canonical date fields and valid ordering."""
        with self.assertRaises(ValidationError):
            MarketDataConfig(name="sp500", version=3, description="Test.", dataset_type="market_data", symbols=["AAPL"], time_range={"start_date": date(2026, 1, 2), "end_date": date(2026, 1, 2)})
        with self.assertRaisesRegex(ValidationError, "start_data"):
            MarketDataConfig(name="sp500", version=3, description="Test.", dataset_type="market_data", symbols=["AAPL"], time_range={"start_data": date(2026, 1, 1), "end_date": date(2026, 1, 2)})



if __name__ == "__main__":
    unittest.main()
