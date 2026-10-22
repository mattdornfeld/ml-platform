"""Behavioral Spark tests for built-in feature and label transforms."""

import os
import sys
import unittest
from datetime import UTC, datetime, timedelta

from pyspark.sql import SparkSession

from ml_platform.data_sdk.feature_transforms import (
    LogReturnTransform,
    RelativeVolumeTransform,
    TrailingVolatilityTransform,
)
from ml_platform.data_sdk.label_transforms import TripleBarrierLabelTransform
from ml_platform.data_sdk.transforms._types import (
    Dataset,
    DatasetReference,
    DataType,
    FeatureKey,
    PartitionType,
)


class BuiltInTransformTest(unittest.TestCase):
    """Verify values and no-future-data behavior of built-in transforms."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start a local UTC Spark session."""
        cls.previous_python = os.environ.get("PYSPARK_PYTHON")
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("built-in-transform-test")
            .getOrCreate()
        )
        cls.spark.conf.set("spark.sql.session.timeZone", "UTC")
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop Spark and restore the previous Python worker setting."""
        cls.spark.stop()
        if cls.previous_python is None:
            os.environ.pop("PYSPARK_PYTHON", None)
        else:
            os.environ["PYSPARK_PYTHON"] = cls.previous_python

    def _market_data(self) -> Dataset:
        """Build deterministic hourly bars for one symbol."""
        start = datetime(2026, 1, 2, 14, tzinfo=UTC)
        rows = [
            (
                start + timedelta(hours=index),
                "AAA",
                100.0 + index * 10.0,
                101.0 + index * 10.0,
                99.0 + index * 10.0,
                100.0 + index * 10.0,
                10 + index * 10,
            )
            for index in range(6)
        ]
        return Dataset(
            metadata=DatasetReference(
                name="market_data",
                version=1,
                num_lookback_partitions=5,
                num_lookahead_partitions=0,
            ),
            frame=self.spark.createDataFrame(
                rows,
                "timestamp timestamp, symbol string, Open double, High double, "
                "Low double, Close double, Volume long",
            ),
        )

    def test_log_return_uses_only_the_previous_close(self) -> None:
        """Log return emits one value per row after the first bar."""
        rows = (
            LogReturnTransform(
                market_data=self._market_data(),
                data_type=DataType.DOUBLE,
                keys=[FeatureKey.SYMBOL],
                partition_type=PartitionType.HOURLY,
            )()
            .orderBy("timestamp")
            .collect()
        )
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(rows[0]["feature_value"], 0.0953101798)

    def test_relative_volume_excludes_the_current_bar_from_history(
        self,
    ) -> None:
        """Relative volume divides each bar by prior bars only."""
        rows = (
            RelativeVolumeTransform(
                market_data=self._market_data(),
                lookback_bars=2,
                data_type=DataType.DOUBLE,
                keys=[FeatureKey.SYMBOL],
                partition_type=PartitionType.HOURLY,
            )()
            .orderBy("timestamp")
            .collect()
        )
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(rows[0]["feature_value"], 2.0)
        self.assertAlmostEqual(rows[-1]["feature_value"], 1.3333333333)

    def test_trailing_volatility_requires_complete_prior_history(self) -> None:
        """Volatility starts only after prior returns fill the requested window."""
        rows = (
            TrailingVolatilityTransform(
                market_data=self._market_data(),
                lookback_bars=2,
                data_type=DataType.DOUBLE,
                keys=[FeatureKey.SYMBOL],
                partition_type=PartitionType.HOURLY,
            )()
            .orderBy("timestamp")
            .collect()
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["feature_value"] > 0 for row in rows))

    def test_triple_barrier_emits_signed_labels(self) -> None:
        """Triple barrier produces labels only for events with a full horizon."""
        rows = (
            TripleBarrierLabelTransform(
                market_data=self._market_data(),
                horizon_bars=1,
                volatility_lookback_bars=2,
                profit_taking_multiplier=0.1,
                stop_loss_multiplier=0.1,
                data_type=DataType.INT,
                keys=[FeatureKey.SYMBOL],
                partition_type=PartitionType.HOURLY,
            )()
            .orderBy("timestamp")
            .collect()
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["label_value"] for row in rows}, {1})
