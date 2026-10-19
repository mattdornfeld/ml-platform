"""Spark and Iceberg coverage for market data and labels."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from pytorch_tutorial.configs import (
    DEFAULT_VARIANT_NAME,
    LongShortLabelConfig,
    MarketDataConfig,
    VariantsConfig,
    load_variants_config,
    parse_variant_args,
    with_variants_config,
)
from pytorch_tutorial.labels import LABEL_TABLE_IDENTIFIER, materialize_long_short_labels
from pytorch_tutorial.market_data import (
    ICEBERG_TABLE_IDENTIFIER,
    create_spark_session,
    fetch_hourly_ohlcv,
    materialize_daily_ohlcv_partitions,
)

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
        cls.spark = create_spark_session(cls.warehouse)
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop Spark and remove the temporary warehouse."""
        cls.spark.stop()
        cls.temporary_directory.cleanup()

    def setUp(self) -> None:
        """Drop tables left by the preceding test."""
        for identifier in (LABEL_TABLE_IDENTIFIER, ICEBERG_TABLE_IDENTIFIER):
            if self.spark.catalog.tableExists(identifier):
                self.spark.sql(f"DROP TABLE {identifier} PURGE")

    @patch("pytorch_tutorial.market_data._request_yahoo_chart")
    def test_yahoo_json_writes_spark_dataframe_without_partition_column(self, request: object) -> None:
        """Convert mocked JSON to Spark and persist only the declared table schema."""
        request.return_value = yahoo_payload()
        frame = fetch_hourly_ohlcv(self.spark, "mock", start_date=TEST_START_DATE, end_date=TEST_END_DATE)
        self.assertNotIn("timestamp_day", frame.columns)
        self.assertEqual(materialize_daily_ohlcv_partitions(frame, self.warehouse), 2)

        rows = self.spark.table(ICEBERG_TABLE_IDENTIFIER).orderBy("timestamp").collect()
        self.assertEqual([row.symbol for row in rows], ["MOCK"] * 3)
        self.assertEqual([row.Close for row in rows], [100.5, 101.5, 102.5])
        self.assertNotIn("timestamp_day", self.spark.table(ICEBERG_TABLE_IDENTIFIER).columns)
        request.assert_called_once_with("MOCK", TEST_START_DATE, TEST_END_DATE)

    def test_merge_and_labels_do_not_write_timestamp_day(self) -> None:
        """Preserve symbol rows and keep time partition fields out of table schemas."""
        first = self.spark.createDataFrame(
            [("2026-01-02T14:30:00Z", "AAA", "regular", 100.0, 101.0, 99.0, 100.5, 1000)],
            "timestamp string, symbol string, market_session string, Open double, High double, Low double, Close double, Volume long",
        )
        bars = first
        for hour, close in enumerate([101.0, 102.0, 104.0, 100.0], start=15):
            bars = bars.unionByName(self.spark.createDataFrame(
                [(f"2026-01-02T{hour}:30:00Z", "AAA", "regular", close, close, close, close, 1000)], first.schema
            ))
        materialize_daily_ohlcv_partitions(bars, self.warehouse)
        labels = LongShortLabelConfig(horizon_bars=1, volatility_lookback_bars=2, profit_taking_multiplier=0.5, stop_loss_multiplier=0.5)
        self.assertGreater(materialize_long_short_labels(self.warehouse, labels, spark=self.spark), 0)
        self.assertNotIn("timestamp_day", self.spark.table(LABEL_TABLE_IDENTIFIER).columns)


class MarketDataUnitTest(unittest.TestCase):
    """Cover configuration and decorated entrypoint behavior."""

    def test_config_rejects_invalid_and_aliased_dates(self) -> None:
        """Require the canonical date fields and valid ordering."""
        with self.assertRaises(ValidationError):
            MarketDataConfig(symbols=["AAPL"], start_date="2026-01-02", end_date="2026-01-02")
        with self.assertRaisesRegex(ValidationError, "start_data"):
            MarketDataConfig(symbols=["AAPL"], start_data="2026-01-01", end_date="2026-01-02")

    def test_variant_decorator_injects_config(self) -> None:
        """Inject the selected typed variant into an entrypoint."""
        received: list[VariantsConfig] = []
        @with_variants_config
        def entrypoint(config: VariantsConfig) -> None:
            received.append(config)
        with patch("sys.argv", ["entrypoint"]):
            entrypoint()
        self.assertEqual(len(received[0].market_data.symbols), 503)
        with patch("sys.argv", ["entrypoint"]):
            self.assertEqual(parse_variant_args().variant, DEFAULT_VARIANT_NAME)
        self.assertEqual(load_variants_config("sp500").labels.horizon_bars, 24)


if __name__ == "__main__":
    unittest.main()
