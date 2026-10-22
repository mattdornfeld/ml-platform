"""Trailing log-return volatility feature transformation."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._types import Dataset
from ml_platform.data_sdk.utils import record


@record
class TrailingVolatilityTransform(FeatureTransform):
    """Compute volatility from a transform-specific number of preceding bars."""

    market_data: Dataset
    lookback_bars: int

    def __call__(self) -> DataFrame:
        """Return sample volatility based exclusively on prior log returns."""
        window = Window.partitionBy("symbol").orderBy("timestamp")
        history = window.rowsBetween(-self.lookback_bars, -1)
        returns = self.market_data.frame.withColumn(
            "previous_close", F.lag("Close").over(window)
        ).withColumn("log_return", F.log(F.try_divide("Close", "previous_close")))
        return returns.select(
            "timestamp",
            "symbol",
            F.stddev_samp("log_return").over(history).alias("feature_value"),
        ).where(F.col("feature_value").isNotNull())
