"""Triple-barrier label strategy."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ml_platform.data_sdk.transforms._label_transform import LabelTransform
from ml_platform.data_sdk.transforms._types import Dataset
from ml_platform.data_sdk.utils import record


@record
class TripleBarrierLabelTransform(LabelTransform):
    """Create volatility-scaled triple-barrier labels from market-data observations."""

    market_data: Dataset
    horizon_bars: int
    volatility_lookback_bars: int
    profit_taking_multiplier: float
    stop_loss_multiplier: float

    def __call__(self) -> DataFrame:
        """Return profit-taking, stop-loss, or timeout labels without future leakage."""
        by_symbol = Window.partitionBy("symbol").orderBy("timestamp")
        volatility_window = by_symbol.rowsBetween(-self.volatility_lookback_bars, -1)
        source = self.market_data.frame.select(
            "timestamp", "symbol", "High", "Low", "Close"
        )
        indexed_observations = source.withColumn(
            "bar_index", F.row_number().over(by_symbol)
        )
        observations = (
            indexed_observations.withColumn(
                "previous_close", F.lag("Close").over(by_symbol)
            )
            .withColumn("log_return", F.log(F.col("Close") / F.col("previous_close")))
            .withColumn(
                "volatility", F.stddev_samp("log_return").over(volatility_window)
            )
            .withColumn(
                "future_bars",
                F.sort_array(
                    F.collect_list(
                        F.struct(
                            F.col("bar_index").alias("future_bar_index"),
                            F.col("timestamp").alias("future_timestamp"),
                            F.col("High").alias("future_high"),
                            F.col("Low").alias("future_low"),
                        )
                    ).over(by_symbol.rowsBetween(1, self.horizon_bars))
                ),
            )
        )
        events = (
            observations.where(F.col("volatility").isNotNull())
            .select(
                F.col("timestamp").alias("event_timestamp"),
                F.col("symbol").alias("event_symbol"),
                F.col("bar_index").alias("event_bar_index"),
                F.col("Close").alias("close"),
                F.col("volatility"),
                F.col("future_bars"),
            )
            .withColumn(
                "upper_barrier",
                F.col("close")
                * (1 + self.profit_taking_multiplier * F.col("volatility")),
            )
            .withColumn(
                "lower_barrier",
                F.col("close") * (1 - self.stop_loss_multiplier * F.col("volatility")),
            )
        )
        profit_hits = F.filter(
            F.col("future_bars"),
            lambda bar: bar["future_high"] >= F.col("upper_barrier"),
        )
        stop_hits = F.filter(
            F.col("future_bars"),
            lambda bar: bar["future_low"] <= F.col("lower_barrier"),
        )
        return (
            events.where(F.size("future_bars") == self.horizon_bars)
            .withColumn("profit_hit", F.try_element_at(profit_hits, F.lit(1)))
            .withColumn("stop_hit", F.try_element_at(stop_hits, F.lit(1)))
            .withColumn(
                "vertical_barrier_timestamp",
                F.element_at("future_bars", -1).getField("future_timestamp"),
            )
            .withColumn(
                "profit_taking_bar_index",
                F.col("profit_hit").getField("future_bar_index"),
            )
            .withColumn(
                "stop_loss_bar_index", F.col("stop_hit").getField("future_bar_index")
            )
            .withColumn("timestamp", F.col("event_timestamp"))
            .withColumn("symbol", F.col("event_symbol"))
            .withColumn("horizon_bars", F.lit(self.horizon_bars).cast("int"))
            .withColumn(
                "barrier_outcome",
                F.when(
                    F.col("profit_taking_bar_index").isNotNull()
                    & (
                        F.col("stop_loss_bar_index").isNull()
                        | (
                            F.col("profit_taking_bar_index")
                            < F.col("stop_loss_bar_index")
                        )
                    ),
                    F.lit("profit_taking"),
                )
                .when(
                    F.col("stop_loss_bar_index").isNotNull()
                    & (
                        F.col("profit_taking_bar_index").isNull()
                        | (
                            F.col("stop_loss_bar_index")
                            < F.col("profit_taking_bar_index")
                        )
                    ),
                    F.lit("stop_loss"),
                )
                .when(
                    F.col("profit_taking_bar_index").isNotNull()
                    & (
                        F.col("profit_taking_bar_index") == F.col("stop_loss_bar_index")
                    ),
                    F.lit("ambiguous"),
                )
                .otherwise(F.lit("timeout")),
            )
            .withColumn(
                "long_short_label",
                F.when(F.col("barrier_outcome") == "profit_taking", F.lit(1))
                .when(F.col("barrier_outcome") == "stop_loss", F.lit(-1))
                .otherwise(F.lit(0))
                .cast("int"),
            )
            .select(
                "timestamp",
                "symbol",
                F.col("long_short_label").alias("label_value"),
            )
        )
