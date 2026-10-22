"""Unit and Spark/Iceberg integration coverage for triple-barrier labels."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from pyspark.sql import SparkSession

from ml_platform.data_sdk.transforms._types import DataType, Dataset, DatasetReference, FeatureKey, PartitionType
from ml_platform.data_sdk.transforms._label_transform import LabelTransform
from ml_platform.data_sdk.label_transforms._triple_barrier import TripleBarrierLabelTransform
from ml_platform.data_sdk.materializers._label import LabelMaterializationConfig, LabelMaterializer
from ml_platform.data_sdk.materializers._dataset import DatasetMaterializer
from ml_platform.data_sdk.materializers._iceberg_table import IcebergTable
from tests.sdk.materializer_support import market_config, materialize_market_data




LABEL_NAME = "test_hourly_triple_barrier"
LABEL_VERSION = 1


def label_table_id() -> str:
    """Return the immutable label table identifier used by this integration test."""
    return IcebergTable.id_for("ml_platform", "labels", LABEL_NAME, LABEL_VERSION)


def market_table_id() -> str:
    """Return the immutable market-data table identifier used by this integration test."""
    return IcebergTable.id_for("ml_platform", "market_data", "sp500", 3)


def triple_barrier_transform(market_data: Dataset) -> TripleBarrierLabelTransform:
    """Build the test strategy with API-delivered parameters."""
    return TripleBarrierLabelTransform(
        keys=[FeatureKey.SYMBOL],
        partition_type=PartitionType.DAILY,
        data_type=DataType.INT,
        market_data=market_data,
        horizon_bars=1,
        volatility_lookback_bars=2,
        profit_taking_multiplier=0.5,
        stop_loss_multiplier=0.5,
    )


class TripleBarrierTransformTest(unittest.TestCase):
    """Exercise the strategy without requiring an Iceberg catalog."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start a local Spark session for the pure DataFrame transformation."""
        cls.previous_python = os.environ.get("PYSPARK_PYTHON")
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.spark = SparkSession.builder.master("local[1]").appName(
            "triple-barrier-transform-test"
        ).config("spark.ui.enabled", "false").getOrCreate()
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop the transform-only Spark session."""
        cls.spark.stop()
        if cls.previous_python is None:
            os.environ.pop("PYSPARK_PYTHON", None)
        else:
            os.environ["PYSPARK_PYTHON"] = cls.previous_python

    def test_returns_triple_barrier_rows_for_injected_market_data(self) -> None:
        """Derive label outcomes solely from the dataset injected into the transform."""
        source = self.spark.createDataFrame(
            [
                ("2026-01-02T14:30:00Z", "AAA", 100.0, 100.0, 100.0),
                ("2026-01-02T15:30:00Z", "AAA", 101.0, 101.0, 101.0),
                ("2026-01-02T16:30:00Z", "AAA", 102.0, 102.0, 102.0),
                ("2026-01-02T17:30:00Z", "AAA", 102.01, 102.01, 102.01),
                ("2026-01-02T18:30:00Z", "AAA", 104.0, 104.0, 104.0),
            ],
            "timestamp string, symbol string, High double, Low double, Close double",
        )
        transform = TripleBarrierLabelTransform(
            market_data=Dataset(
                DatasetReference(name="sp500", version=3, num_lookback_partitions=0), source
            ),
            keys=[FeatureKey.SYMBOL],
            partition_type=PartitionType.DAILY,
            data_type=DataType.INT,
            horizon_bars=1,
            volatility_lookback_bars=2,
            profit_taking_multiplier=0.5,
            stop_loss_multiplier=0.5,
        )

        rows = transform().orderBy("timestamp").collect()

        self.assertTrue(rows)
        self.assertEqual(
            set(rows[0].asDict()),
            {"timestamp", "symbol", "label_value"},
        )
        self.assertIn(rows[0].label_value, {-1, 0, 1})


class LabelsIntegrationTest(unittest.TestCase):
    """Verify labels are derived and persisted from a synthetic market table."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start Spark with a disposable Iceberg warehouse."""
        cls.temporary_directory = TemporaryDirectory()
        cls.warehouse = Path(cls.temporary_directory.name) / "warehouse"
        cls.spark = DatasetMaterializer._create_spark_session(cls.warehouse)
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop Spark and clean up the warehouse."""
        cls.spark.stop()
        cls.temporary_directory.cleanup()

    def setUp(self) -> None:
        """Create a source market table with known next-bar outcomes."""
        for identifier in (label_table_id(), market_table_id()):
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
        materialize_market_data(self.spark, bars, market_config(self.warehouse))

    def _label_worker(self) -> LabelMaterializer:
        """Construct the runtime label definition used by direct Spark writes."""
        config = LabelMaterializationConfig.model_validate_json(json.dumps({
            "name": LABEL_NAME,
            "version": LABEL_VERSION,
            "description": "Synthetic triple-barrier labels.",
            "warehouse_dir": str(self.warehouse),
            "datasets": [{"name": "sp500", "version": 3, "dataset_type": "market_data", "argument_name": "market_data", "num_lookback_partitions": 0}],
            "transform": "ml_platform.data_sdk.label_transforms._triple_barrier.TripleBarrierLabelTransform",
            "parameters": {"horizon_bars": 1, "volatility_lookback_bars": 2,
                           "profit_taking_multiplier": 0.5, "stop_loss_multiplier": 0.5},
            "partitions": [{
                "partition_id": "synthetic-label-day",
                "timestamp": "2026-01-02T00:00:00Z",
                "callback_token": "synthetic-label-token",
                "partition_dependencies": [{"partition_id": "synthetic-market-day", "dataset_name": "sp500",
                                        "dataset_version": 3, "timestamp": "2026-01-02T00:00:00Z", "snapshot_id": 1}],
            }],
        }))
        return LabelMaterializer(config)

    def _write_labels(self) -> int:
        """Materialize the test label output through the shared lifecycle."""
        worker = self._label_worker()
        transform = triple_barrier_transform(Dataset(
            DatasetReference(name="sp500", version=3, num_lookback_partitions=0),
            self.spark.table(market_table_id()),
        ))
        requested_partitions = worker._requested_partitions(self.spark)
        labels = worker._filter_partitions(transform(), requested_partitions)
        worker.validate(labels)
        worker.iceberg_table.create(self.spark)
        worker.iceberg_table.write(labels)
        return labels.count()

    def test_writes_expected_outcomes_and_is_idempotent(self) -> None:
        """Materialize profit, stop, and timeout labels without future leakage."""
        first_count = self._write_labels()
        rows = self.spark.table(label_table_id()).orderBy("timestamp").collect()
        outcomes = {row.label_value for row in rows}

        self.assertGreaterEqual(first_count, 3)
        self.assertTrue({-1, 0, 1} <= outcomes)
        self.assertNotIn("timestamp_day", self.spark.table(label_table_id()).columns)
        self.assertEqual(
            self._write_labels(),
            first_count,
        )
        self.assertEqual(self.spark.table(label_table_id()).count(), len(rows))
        self.assertEqual(
            self._label_worker().iceberg_table.storage_path(self.warehouse),
            self.warehouse.resolve() / "labels" / "test_hourly_triple_barrier-v1",
        )
        self.assertEqual(
            sorted(path.name for path in (self._label_worker().iceberg_table.storage_path(self.warehouse) / "data").glob("timestamp_day=*")),
            ["timestamp_day=2026-01-02"],
        )


class LabelsUnitTest(unittest.TestCase):
    """Cover validation and entrypoint behavior without a Spark runtime."""

    def test_triple_barrier_strategy_implements_label_transform(self) -> None:
        """Expose the strategy behind the common label-transform interface."""
        self.assertTrue(issubclass(TripleBarrierLabelTransform, LabelTransform))




if __name__ == "__main__":
    unittest.main()
