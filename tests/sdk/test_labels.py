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
from ml_platform.data_sdk.materializers._label import LabelMaterializationConfig, LabelMaterializer, LabelResource
from ml_platform.data_sdk.materializers._dataset import DatasetMaterializer
from tests.sdk.materializer_support import market_config



def label_resource() -> LabelResource:
    """Return the API-delivered identity corresponding to the test definition."""
    return LabelResource(name="test_hourly_triple_barrier", version=1)


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
        for identifier in (LabelMaterializer._table_identifier_for(label_resource()), DatasetMaterializer._iceberg_table_identifier('sp500', 3)):
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
        DatasetMaterializer._materialize_test_frame(
            bars, config=market_config(self.warehouse), spark=self.spark
        )

    def _label_worker(self) -> LabelMaterializer:
        """Construct the runtime label definition used by direct Spark writes."""
        config = LabelMaterializationConfig.model_validate_json(json.dumps({
            "name": label_resource().name,
            "version": label_resource().version,
            "description": "Synthetic triple-barrier labels.",
            "warehouse_dir": str(self.warehouse),
            "datasets": [{"name": "sp500", "version": 3, "num_lookback_partitions": 0}],
            "transform": "ml_platform.data_sdk.label_transforms._triple_barrier.TripleBarrierLabelTransform",
            "parameters": {"horizon_bars": 1, "volatility_lookback_bars": 2,
                           "profit_taking_multiplier": 0.5, "stop_loss_multiplier": 0.5},
            "partitions": [{
                "partition_id": "synthetic-label-day",
                "timestamp": "2026-01-02T00:00:00Z",
                "callback_token": "synthetic-label-token",
                "dataset_partitions": [{"partition_id": "synthetic-market-day", "dataset_name": "sp500",
                                        "dataset_version": 3, "timestamp": "2026-01-02T00:00:00Z", "snapshot_id": 1}],
            }],
        }))
        return LabelMaterializer(config, datasets={})

    def test_writes_expected_outcomes_and_is_idempotent(self) -> None:
        """Materialize profit, stop, and timeout labels without future leakage."""
        first_count = self._label_worker()._materialize_transform(
            triple_barrier_transform(
                Dataset(
                    DatasetReference(name="sp500", version=3, num_lookback_partitions=0),
                    self.spark.table(DatasetMaterializer._iceberg_table_identifier('sp500', 3)),
                )
            ),
            spark=self.spark,
        )
        rows = self.spark.table(LabelMaterializer._table_identifier_for(label_resource())).orderBy("timestamp").collect()
        outcomes = {row.label_value for row in rows}

        self.assertGreaterEqual(first_count, 3)
        self.assertTrue({-1, 0, 1} <= outcomes)
        self.assertNotIn("timestamp_day", self.spark.table(LabelMaterializer._table_identifier_for(label_resource())).columns)
        self.assertEqual(
            self._label_worker()._materialize_transform(
            triple_barrier_transform(
                Dataset(
                    DatasetReference(name="sp500", version=3, num_lookback_partitions=0),
                    self.spark.table(DatasetMaterializer._iceberg_table_identifier('sp500', 3)),
                )
            ),
            spark=self.spark,
        ),
            first_count,
        )
        self.assertEqual(self.spark.table(LabelMaterializer._table_identifier_for(label_resource())).count(), len(rows))
        self.assertEqual(
            LabelMaterializer._storage_path(self.warehouse, label_resource()),
            self.warehouse.resolve() / "labels" / "test_hourly_triple_barrier-v1",
        )
        self.assertEqual(
            sorted(path.name for path in (LabelMaterializer._storage_path(self.warehouse, label_resource()) / "data").glob("timestamp_day=*")),
            ["timestamp_day=2026-01-02"],
        )


class LabelsUnitTest(unittest.TestCase):
    """Cover validation and entrypoint behavior without a Spark runtime."""

    def test_triple_barrier_strategy_implements_label_transform(self) -> None:
        """Expose the strategy behind the common label-transform interface."""
        self.assertTrue(issubclass(TripleBarrierLabelTransform, LabelTransform))




if __name__ == "__main__":
    unittest.main()
