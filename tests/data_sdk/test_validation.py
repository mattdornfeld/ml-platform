"""Spark validation behavior shared by materializer types."""

import os
import sys
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from pyspark.sql import SparkSession
from pyspark.sql import functions as spark_functions

from ml_platform.data_sdk.dataset_transforms._market_data import (
    MarketDataTransform,
)
from ml_platform.data_sdk.exceptions import (
    FiniteValuesValidationError,
    InputFeatureKeysValidationError,
    RequestedPartitionsValidationError,
)
from ml_platform.data_sdk.materializers import (
    ExactColumnsValidationConfig,
    MaterializationValidator,
    MaterializerPartition,
    MinimumRowsValidation,
    ValidationConfigs,
)
from ml_platform.data_sdk.materializers._dataset import (
    DatasetMaterializationConfig,
    DatasetMaterializer,
)
from ml_platform.data_sdk.materializers._feature import (
    FeatureMaterializationConfig,
    FeatureMaterializer,
)
from ml_platform.data_sdk.transforms._types import (
    DatasetType,
    FeatureKey,
    PartitionType,
)


class MaterializerValidationTest(unittest.TestCase):
    """Exercise shared checks with real PySpark DataFrames."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_python = os.environ.get("PYSPARK_PYTHON")
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("materializer-validation-test")
            .getOrCreate()
        )
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
            partitions=[datetime(2026, 1, 1, tzinfo=UTC)],
            keys=["symbol"],
            partition_type="daily",
        )

    def test_xnys_regular_session_filter_selects_sessions_and_skips_closed_periods(
        self,
    ) -> None:
        """XNYS recognizes holidays, regular session hours, and post-close hours."""
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        self.assertFalse(
            MinimumRowsValidation._overlaps_xnys_regular_session(
                datetime(2026, 1, 1, tzinfo=UTC), timedelta(days=1), calendar
            )
        )
        self.assertTrue(
            MinimumRowsValidation._overlaps_xnys_regular_session(
                datetime(2026, 1, 2, tzinfo=UTC), timedelta(days=1), calendar
            )
        )
        self.assertTrue(
            MinimumRowsValidation._overlaps_xnys_regular_session(
                datetime(2026, 1, 2, 14, tzinfo=UTC),
                timedelta(hours=1),
                calendar,
            )
        )
        self.assertFalse(
            MinimumRowsValidation._overlaps_xnys_regular_session(
                datetime(2026, 1, 2, 21, tzinfo=UTC),
                timedelta(hours=1),
                calendar,
            )
        )

    def test_validator_runs_only_the_injected_dataframe_validations(
        self,
    ) -> None:
        """The shared validator has no dependency on any materializer resource type."""
        frame = self.spark.createDataFrame([("AAA",)], "symbol string")
        validator = MaterializationValidator(
            ValidationConfigs(
                default_validation_configs=(
                    ExactColumnsValidationConfig(
                        reason="Output must contain only the configured columns.",
                        columns=("symbol",),
                    ),
                ),
                override_validation_configs=(),
            )
        )

        self.assertIsNone(validator.validate(frame))
        with self.assertRaisesRegex(
            ValueError, "Output must contain only the configured columns"
        ):
            validator.validate(frame.selectExpr("symbol", "1 AS unexpected"))

    def test_dataset_rejects_duplicate_and_unrequested_rows(self) -> None:
        """Market rows may leave requested days empty but cannot mutate other days."""
        requested = MaterializerPartition(
            partition_id="day",
            timestamp=datetime(2026, 1, 3, tzinfo=UTC),
            callback_token="token",
        )
        worker = DatasetMaterializer(
            DatasetMaterializationConfig(
                name="market",
                version=1,
                description="Test market data.",
                warehouse_dir="warehouse",
                dataset_type=DatasetType.MARKET_DATA,
                keys=[FeatureKey.SYMBOL],
                partition_type=PartitionType.DAILY,
                symbols=["AAA"],
                validations=[],
                partitions=[requested],
            )
        )
        source = self.spark.createDataFrame(
            [
                (
                    "2026-01-03 15:30:00",
                    "AAA",
                    "regular",
                    100.0,
                    101.0,
                    99.0,
                    100.5,
                    1000,
                )
            ],
            "timestamp string, symbol string, market_session string, Open double, High double, "
            "Low double, Close double, Volume long",
        )
        prepared = source.selectExpr(
            "cast(timestamp as timestamp) as timestamp",
            "symbol",
            "market_session",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        )
        self.assertIsNone(worker.validate(prepared))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            worker.validate(prepared.unionByName(prepared))
        next_day = source.withColumn(
            "timestamp", spark_functions.lit("2026-01-04 15:30:00")
        )
        with self.assertRaisesRegex(
            RequestedPartitionsValidationError, "outside requested"
        ):
            worker.validate(
                next_day.selectExpr(
                    "cast(timestamp as timestamp) as timestamp",
                    "symbol",
                    "market_session",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                )
            )

    def test_feature_specific_value_and_input_key_policies(self) -> None:
        """Feature policy checks remain enforced after shared structural validation."""
        config = FeatureMaterializationConfig.model_validate(
            {
                "name": "returns",
                "version": 1,
                "description": "Test feature.",
                "warehouse_dir": "temporary-warehouse",
                "dtype": "double",
                "frequency": "daily",
                "datasets": [
                    {
                        "name": "sp500",
                        "version": 3,
                        "dataset_type": DatasetType.MARKET_DATA,
                        "argument_name": "market_data",
                        "input_window": {"lookback_partitions": 1},
                    }
                ],
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
                "partitions": [
                    {
                        "partition_id": "output",
                        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                        "callback_token": "token",
                        "partition_dependencies": [
                            {
                                "partition_id": "input",
                                "dataset_name": "sp500",
                                "dataset_version": 3,
                                "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                                "snapshot_id": 1,
                            }
                        ],
                    }
                ],
            }
        )
        source = self.spark.createDataFrame([("AAA",)], "symbol string")
        source.createOrReplaceTempView("input_market_data")

        def feature(symbol: str, value: float):
            return self.spark.createDataFrame(
                [("2026-01-01", symbol, value)],
                "timestamp string, symbol string, feature_value double",
            )

        worker = FeatureMaterializer(config)

        def validate_feature(frame: object) -> None:
            with patch.object(
                FeatureMaterializer,
                "_dataset_table_identifier",
                return_value="input_market_data",
            ):
                worker.validate(frame)

        accepted = feature("AAA", 1.0).selectExpr(
            "cast(timestamp as timestamp) as timestamp",
            "symbol",
            "feature_value",
        )
        self.assertIsNone(validate_feature(accepted))
        with self.assertRaisesRegex(
            FiniteValuesValidationError, "NaN, infinity"
        ):
            validate_feature(
                feature("AAA", float("nan")).selectExpr(
                    "cast(timestamp as timestamp) as timestamp",
                    "symbol",
                    "feature_value",
                )
            )
        with self.assertRaisesRegex(
            InputFeatureKeysValidationError, "absent from input"
        ):
            validate_feature(
                feature("BBB", 1.0).selectExpr(
                    "cast(timestamp as timestamp) as timestamp",
                    "symbol",
                    "feature_value",
                )
            )
        uncastable = self.spark.createDataFrame(
            [("2026-01-01", "AAA", "invalid")],
            "timestamp string, symbol string, feature_value string",
        )
        with self.assertRaisesRegex(
            FiniteValuesValidationError, "NaN, infinity"
        ):
            validate_feature(
                uncastable.selectExpr(
                    "cast(timestamp as timestamp) as timestamp",
                    "symbol",
                    "feature_value",
                )
            )

    def test_feature_partition_filter_respects_hourly_frequency(self) -> None:
        """Two requested hours replace two hourly partitions, not their entire day."""
        callbacks = [
            MaterializerPartition(
                partition_id=f"output-{hour}",
                timestamp=datetime(2026, 1, 1, hour, tzinfo=UTC),
                callback_token="token",
            )
            for hour in (10, 11)
        ]
        worker = FeatureMaterializer(
            SimpleNamespace(
                partition_type=PartitionType.HOURLY,
                partition_timestamps=tuple(
                    callback.timestamp for callback in callbacks
                ),
                partitions=callbacks,
                feature_keys=["symbol"],
            )
        )
        source = self.spark.createDataFrame(
            [
                ("2026-01-01 10:30:00", "AAA", 1.0),
                ("2026-01-01 11:30:00", "AAA", 2.0),
                ("2026-01-01 12:30:00", "AAA", 3.0),
            ],
            "timestamp string, symbol string, feature_value double",
        ).selectExpr(
            "cast(timestamp as timestamp) as timestamp",
            "symbol",
            "feature_value",
        )

        requested = worker._requested_partitions(self.spark)
        selected = worker._filter_partitions(source, requested)

        self.assertEqual(selected.count(), 2)
