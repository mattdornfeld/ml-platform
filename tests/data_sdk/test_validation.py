"""Spark validation behavior shared by materializer types."""

from datetime import datetime, timedelta, timezone
import os
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from pyspark.sql import SparkSession, functions as F

from ml_platform.data_sdk.materializers import (
    ExactColumnsValidationConfig,
    MaterializerPartition,
    MaterializationValidator,
    ValidationConfigs,
    MinimumRowsValidationConfig,
    FeatureKeysFromInputValidationConfig,
    XnysRegularSessionTemporalFilterConfig,
    MinimumRowsValidation,
)
from ml_platform.data_sdk.materializers._dataset import DatasetMaterializationConfig, DatasetMaterializer
from ml_platform.data_sdk.dataset_transforms._market_data import MarketDataTransform
from ml_platform.data_sdk.transforms._types import FeatureKey, PartitionType
from ml_platform.data_sdk.materializers._feature import FeatureMaterializationConfig, FeatureMaterializer


class MaterializerValidationTest(unittest.TestCase):
    """Exercise shared checks with real PySpark DataFrames."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_python = os.environ.get("PYSPARK_PYTHON")
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.spark = SparkSession.builder.master("local[1]").appName("materializer-validation-test").getOrCreate()
        cls.spark.conf.set("spark.sql.session.timeZone", "UTC")
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()
        if cls.previous_python is None:
            os.environ.pop("PYSPARK_PYTHON", None)
        else:
            os.environ["PYSPARK_PYTHON"] = cls.previous_python

    def _daily_dataset_transform(self) -> MarketDataTransform:
        """Return the static transform layout required by dataset validation tests."""
        return MarketDataTransform(
            spark=self.spark,
            symbols=["AAA"],
            partitions=[datetime(2026, 1, 1, tzinfo=timezone.utc)],
            keys=["symbol"],
            partition_type="daily",
        )


    def test_xnys_regular_session_filter_selects_sessions_and_skips_closed_periods(self) -> None:
        """XNYS recognizes holidays, regular session hours, and post-close hours."""
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        self.assertFalse(
            MinimumRowsValidation._overlaps_xnys_regular_session(
                datetime(2026, 1, 1, tzinfo=timezone.utc), timedelta(days=1), calendar
            )
        )
        self.assertTrue(
            MinimumRowsValidation._overlaps_xnys_regular_session(
                datetime(2026, 1, 2, tzinfo=timezone.utc), timedelta(days=1), calendar
            )
        )
        self.assertTrue(
            MinimumRowsValidation._overlaps_xnys_regular_session(
                datetime(2026, 1, 2, 14, tzinfo=timezone.utc), timedelta(hours=1), calendar
            )
        )
        self.assertFalse(
            MinimumRowsValidation._overlaps_xnys_regular_session(
                datetime(2026, 1, 2, 21, tzinfo=timezone.utc), timedelta(hours=1), calendar
            )
        )

    def test_validator_runs_only_the_injected_dataframe_validations(self) -> None:
        """The shared validator has no dependency on any materializer resource type."""
        frame = self.spark.createDataFrame([("AAA",)], "symbol string")
        validator = MaterializationValidator(
            ValidationConfigs(
                default_validation_configs=(ExactColumnsValidationConfig(
                    reason="Output must contain only the configured columns.",
                    columns=("symbol",),
                ),),
                override_validation_configs=(),
            )
        )

        self.assertIsNone(validator.validate(frame))
        with self.assertRaisesRegex(ValueError, "Output must contain only the configured columns"):
            validator.validate(frame.selectExpr("symbol", "1 AS unexpected"))

    def test_dataset_rejects_duplicate_and_unrequested_rows(self) -> None:
        """Market rows may leave requested days empty but cannot mutate other days."""
        requested = MaterializerPartition(
            partition_id="day", timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
            callback_token="token",
        )
        worker = DatasetMaterializer(DatasetMaterializationConfig(
            name="market", version=1, description="Test market data.", warehouse_dir="warehouse",
            dataset_type="market_data", keys=[FeatureKey.SYMBOL], partition_type=PartitionType.DAILY, symbols=["AAA"],
            validations=[], partitions=[requested],
        ))
        source = self.spark.createDataFrame(
            [("2026-01-03 15:30:00", "AAA", "regular", 100.0, 101.0, 99.0, 100.5, 1000)],
            "timestamp string, symbol string, market_session string, Open double, High double, "
            "Low double, Close double, Volume long",
        )
        prepared = worker._prepare_frame(source)
        self.assertIsNone(worker.validate(prepared))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            worker.validate(prepared.unionByName(prepared))
        next_day = source.withColumn("timestamp", F.lit("2026-01-04 15:30:00"))
        with self.assertRaisesRegex(ValueError, "requested partitions"):
            worker.validate(worker._prepare_frame(next_day))

    def test_feature_specific_value_and_input_key_policies(self) -> None:
        """Feature policy checks remain enforced after shared structural validation."""
        config = FeatureMaterializationConfig.model_validate({
            "name": "returns", "version": 1, "description": "Test feature.",
            "warehouse_dir": "temporary-warehouse", "dtype": "double", "frequency": "daily",
            "datasets": [{"name": "sp500", "version": 3, "argument_name": "market_data", "num_lookback_partitions": 1}],
            "transform": "ml_platform.data_sdk.LogReturnTransform",
            "validations": [
                {
                    "type": "minimum_rows",
                    "minimum_rows": 1,
                    "reason": "Each requested feature partition requires a row.",
                },
                {
                    "type": "feature_keys_from_input",
                    "reason": "Feature keys must occur in source market data.",
                },
            ],
            "partitions": [{
                "partition_id": "output", "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc), "callback_token": "token",
                "partition_dependencies": [{
                    "partition_id": "input", "dataset_name": "sp500", "dataset_version": 3,
                    "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc), "snapshot_id": 1,
                }],
            }],
        })
        source = self.spark.createDataFrame([("AAA",)], "symbol string")
        source.createOrReplaceTempView("input_market_data")

        def feature(symbol: str, value: float):
            return self.spark.createDataFrame(
                [("2026-01-01", symbol, value)], "timestamp string, symbol string, feature_value double"
            )

        worker = FeatureMaterializer(config)
        def validate_feature(frame: object) -> None:
            with patch.object(FeatureMaterializer, "_dataset_table_identifier", return_value="input_market_data"):
                worker.validate(frame)

        accepted = worker._prepare_frame(feature("AAA", 1.0))
        self.assertIsNone(validate_feature(accepted))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_feature(worker._prepare_frame(feature("AAA", float("nan"))))
        with self.assertRaisesRegex(ValueError, "absent from input"):
            validate_feature(worker._prepare_frame(feature("BBB", 1.0)))
        uncastable = self.spark.createDataFrame(
            [("2026-01-01", "AAA", "invalid")],
            "timestamp string, symbol string, feature_value string",
        )
        with self.assertRaisesRegex(ValueError, "non-null, castable"):
            validate_feature(worker._prepare_frame(uncastable))

    def test_feature_partition_filter_respects_hourly_frequency(self) -> None:
        """Two requested hours replace two hourly partitions, not their entire day."""
        callbacks = [
            MaterializerPartition(
                partition_id=f"output-{hour}", timestamp=datetime(2026, 1, 1, hour, tzinfo=timezone.utc),
                callback_token="token",
            )
            for hour in (10, 11)
        ]
        worker = FeatureMaterializer(SimpleNamespace(frequency="hourly", partitions=callbacks, feature_keys=["symbol"]))
        source = self.spark.createDataFrame(
            [("2026-01-01 10:30:00", "AAA", 1.0),
             ("2026-01-01 11:30:00", "AAA", 2.0),
             ("2026-01-01 12:30:00", "AAA", 3.0)],
            "timestamp string, symbol string, feature_value double",
        ).selectExpr("cast(timestamp as timestamp) as timestamp", "symbol", "feature_value")

        requested = worker._requested_partitions(self.spark, "hours(timestamp)")
        selected = worker._filter_partitions(source, requested, "hours(timestamp)")

        self.assertEqual(selected.count(), 2)
        self.assertEqual(worker._target_partition_keys("hours(timestamp)"), ("2026-01-01T10:00:00", "2026-01-01T11:00:00"))
