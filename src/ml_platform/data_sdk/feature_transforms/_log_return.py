"""One-bar close-to-close log-return feature transformation."""

from dataclasses import dataclass

from pyspark.sql import DataFrame, Window, functions as F

from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._types import Dataset


@dataclass(frozen=True)
class LogReturnTransform(FeatureTransform):
    """Compute each symbol's one-bar close-to-close log return."""

    market_data: Dataset

    def __call__(self) -> DataFrame:
        """Return current log returns without using future bars."""
        window = Window.partitionBy("symbol").orderBy("timestamp")
        return self.market_data.frame.withColumn("previous_close", F.lag("Close").over(window)).select(
            "timestamp", "symbol", F.log(F.try_divide("Close", "previous_close")).alias("feature_value")
        ).where(F.col("feature_value").isNotNull())
