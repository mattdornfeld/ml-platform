"""Trailing relative-volume feature transformation."""

from dataclasses import dataclass

from pyspark.sql import DataFrame, Window, functions as F

from features.base import FeatureTransform
from sdk.datasets import Dataset


@dataclass(frozen=True)
class RelativeVolumeTransform(FeatureTransform):
    """Compute volume relative to the declared trailing input-partition window."""

    market_data: Dataset
    lookback_bars: int

    def __call__(self) -> DataFrame:
        """Return current volume divided by the mean of prior bars only."""
        history = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-self.lookback_bars, -1)
        return self.market_data.frame.select(
            "timestamp",
            "symbol",
            F.try_divide("Volume", F.avg("Volume").over(history)).alias("feature_value"),
        ).where(F.col("feature_value").isNotNull())
