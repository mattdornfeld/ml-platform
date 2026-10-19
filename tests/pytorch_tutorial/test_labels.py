"""Unit and Spark/Iceberg integration coverage for triple-barrier labels."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from pytorch_tutorial.configs import LongShortLabelConfig
from pytorch_tutorial.labels import (
    DATA_ROOT,
    LABEL_TABLE_IDENTIFIER,
    main,
    materialize_long_short_labels,
)
from pytorch_tutorial.market_data import (
    ICEBERG_TABLE_IDENTIFIER,
    create_spark_session,
    materialize_daily_ohlcv_partitions,
)


def label_config() -> LongShortLabelConfig:
    """Return a small deterministic triple-barrier configuration."""
    return LongShortLabelConfig(
        horizon_bars=1,
        volatility_lookback_bars=2,
        profit_taking_multiplier=0.5,
        stop_loss_multiplier=0.5,
    )


class LabelsIntegrationTest(unittest.TestCase):
    """Verify labels are derived and persisted from a synthetic market table."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start Spark with a disposable Iceberg warehouse."""
        cls.temporary_directory = TemporaryDirectory()
        cls.warehouse = Path(cls.temporary_directory.name) / "warehouse"
        cls.spark = create_spark_session(cls.warehouse)
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop Spark and clean up the warehouse."""
        cls.spark.stop()
        cls.temporary_directory.cleanup()

    def setUp(self) -> None:
        """Create a source market table with known next-bar outcomes."""
        for identifier in (LABEL_TABLE_IDENTIFIER, ICEBERG_TABLE_IDENTIFIER):
            if self.spark.catalog.tableExists(identifier):
                self.spark.sql(f"DROP TABLE {identifier} PURGE")
        source = self.spark.createDataFrame(
            [
                ("2026-01-02T14:30:00Z", "AAA", "regular", 100.0),
                ("2026-01-02T15:30:00Z", "AAA", "regular", 101.0),
                ("2026-01-02T16:30:00Z", "AAA", "regular", 102.0),
                ("2026-01-02T17:30:00Z", "AAA", "regular", 102.01),
                ("2026-01-02T18:30:00Z", "AAA", "regular", 104.0),
                ("2026-01-02T19:30:00Z", "AAA", "regular", 100.0),
                ("2026-01-02T20:30:00Z", "AAA", "regular", 100.0),
            ],
            "timestamp string, symbol string, market_session string, Close double",
        )
        bars = source.selectExpr(
            "timestamp", "symbol", "market_session", "Close AS Open", "Close AS High",
            "Close AS Low", "Close", "CAST(1000 AS BIGINT) AS Volume",
        )
        materialize_daily_ohlcv_partitions(bars, self.warehouse, spark=self.spark)

    def test_writes_expected_outcomes_and_is_idempotent(self) -> None:
        """Materialize profit, stop, and timeout labels without future leakage."""
        first_count = materialize_long_short_labels(self.warehouse, label_config(), spark=self.spark)
        rows = self.spark.table(LABEL_TABLE_IDENTIFIER).orderBy("timestamp").collect()
        outcomes = {(row.barrier_outcome, row.long_short_label) for row in rows}

        self.assertGreaterEqual(first_count, 3)
        self.assertTrue({("profit_taking", 1), ("stop_loss", -1), ("timeout", 0)} <= outcomes)
        self.assertNotIn("timestamp_day", self.spark.table(LABEL_TABLE_IDENTIFIER).columns)
        self.assertEqual(
            materialize_long_short_labels(self.warehouse, label_config(), spark=self.spark),
            first_count,
        )
        self.assertEqual(self.spark.table(LABEL_TABLE_IDENTIFIER).count(), len(rows))


class LabelsUnitTest(unittest.TestCase):
    """Cover validation and entrypoint behavior without a Spark runtime."""

    def test_label_config_rejects_invalid_parameters(self) -> None:
        """Require positive barriers, a horizon, and enough history."""
        for field, value in (
            ("horizon_bars", 0),
            ("volatility_lookback_bars", 1),
            ("profit_taking_multiplier", 0),
            ("stop_loss_multiplier", -1),
        ):
            values = label_config().model_dump()
            values[field] = value
            with self.assertRaises(ValidationError):
                LongShortLabelConfig(**values)

    def test_rejects_missing_market_table_before_running_spark_plan(self) -> None:
        """Fail clearly when labels have no source market data."""
        spark = MagicMock()
        spark.catalog.tableExists.return_value = False
        with self.assertRaisesRegex(ValueError, "Market-data table does not exist"):
            materialize_long_short_labels(Path("unused"), label_config(), spark=spark)

    @patch("pytorch_tutorial.labels.materialize_long_short_labels", return_value=12)
    def test_entrypoint_uses_default_variant(self, materialize: MagicMock) -> None:
        """Load the default variant and pass only its label config to the writer."""
        with patch("sys.argv", ["labels"]):
            main()
        warehouse, config = materialize.call_args.args
        self.assertEqual(warehouse, DATA_ROOT)
        self.assertEqual(config.horizon_bars, 24)


if __name__ == "__main__":
    unittest.main()
