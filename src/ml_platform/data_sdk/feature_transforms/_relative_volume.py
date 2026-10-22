"""Trailing relative-volume feature transformation."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as spark_functions

from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._types import Dataset
from ml_platform.data_sdk.utils import record


@record
class RelativeVolumeTransform(FeatureTransform):
    """Compute volume relative to the declared trailing input-partition window."""

    market_data: Dataset
    lookback_bars: int

    def __call__(self) -> DataFrame:
        """Return current volume divided by the mean of prior bars only."""
        history = (
            Window.partitionBy("symbol")
            .orderBy("timestamp")
            .rowsBetween(-self.lookback_bars, -1)
        )
        return self.market_data.frame.select(
            "timestamp",
            "symbol",
            spark_functions.try_divide(
                "Volume", spark_functions.avg("Volume").over(history)
            ).alias("feature_value"),
        ).where(spark_functions.col("feature_value").isNotNull())
