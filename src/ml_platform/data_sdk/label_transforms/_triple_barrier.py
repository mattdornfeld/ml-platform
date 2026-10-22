"""Triple-barrier label strategy."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as spark_functions

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
        volatility_window = by_symbol.rowsBetween(
            -self.volatility_lookback_bars, -1
        )
        source = self.market_data.frame.select(
            "timestamp", "symbol", "High", "Low", "Close"
        )
        indexed_observations = source.withColumn(
            "bar_index", spark_functions.row_number().over(by_symbol)
        )
        observations = (
            indexed_observations.withColumn(
                "previous_close", spark_functions.lag("Close").over(by_symbol)
            )
            .withColumn(
                "log_return",
                spark_functions.log(
                    spark_functions.col("Close")
                    / spark_functions.col("previous_close")
                ),
            )
            .withColumn(
                "volatility",
                spark_functions.stddev_samp("log_return").over(
                    volatility_window
                ),
            )
            .withColumn(
                "future_bars",
                spark_functions.sort_array(
                    spark_functions.collect_list(
                        spark_functions.struct(
                            spark_functions.col("bar_index").alias(
                                "future_bar_index"
                            ),
                            spark_functions.col("timestamp").alias(
                                "future_timestamp"
                            ),
                            spark_functions.col("High").alias("future_high"),
                            spark_functions.col("Low").alias("future_low"),
                        )
                    ).over(by_symbol.rowsBetween(1, self.horizon_bars))
                ),
            )
        )
        events = (
            observations.where(spark_functions.col("volatility").isNotNull())
            .select(
                spark_functions.col("timestamp").alias("event_timestamp"),
                spark_functions.col("symbol").alias("event_symbol"),
                spark_functions.col("bar_index").alias("event_bar_index"),
                spark_functions.col("Close").alias("close"),
                spark_functions.col("volatility"),
                spark_functions.col("future_bars"),
            )
            .withColumn(
                "upper_barrier",
                spark_functions.col("close")
                * (
                    1
                    + self.profit_taking_multiplier
                    * spark_functions.col("volatility")
                ),
            )
            .withColumn(
                "lower_barrier",
                spark_functions.col("close")
                * (
                    1
                    - self.stop_loss_multiplier
                    * spark_functions.col("volatility")
                ),
            )
        )
        profit_hits = spark_functions.filter(
            spark_functions.col("future_bars"),
            lambda bar: (
                bar["future_high"] >= spark_functions.col("upper_barrier")
            ),
        )
        stop_hits = spark_functions.filter(
            spark_functions.col("future_bars"),
            lambda bar: (
                bar["future_low"] <= spark_functions.col("lower_barrier")
            ),
        )
        return (
            events.where(
                spark_functions.size("future_bars") == self.horizon_bars
            )
            .withColumn(
                "profit_hit",
                spark_functions.try_element_at(
                    profit_hits, spark_functions.lit(1)
                ),
            )
            .withColumn(
                "stop_hit",
                spark_functions.try_element_at(
                    stop_hits, spark_functions.lit(1)
                ),
            )
            .withColumn(
                "vertical_barrier_timestamp",
                spark_functions.element_at("future_bars", -1).getField(
                    "future_timestamp"
                ),
            )
            .withColumn(
                "profit_taking_bar_index",
                spark_functions.col("profit_hit").getField("future_bar_index"),
            )
            .withColumn(
                "stop_loss_bar_index",
                spark_functions.col("stop_hit").getField("future_bar_index"),
            )
            .withColumn("timestamp", spark_functions.col("event_timestamp"))
            .withColumn("symbol", spark_functions.col("event_symbol"))
            .withColumn(
                "horizon_bars",
                spark_functions.lit(self.horizon_bars).cast("int"),
            )
            .withColumn(
                "barrier_outcome",
                spark_functions.when(
                    spark_functions.col("profit_taking_bar_index").isNotNull()
                    & (
                        spark_functions.col("stop_loss_bar_index").isNull()
                        | (
                            spark_functions.col("profit_taking_bar_index")
                            < spark_functions.col("stop_loss_bar_index")
                        )
                    ),
                    spark_functions.lit("profit_taking"),
                )
                .when(
                    spark_functions.col("stop_loss_bar_index").isNotNull()
                    & (
                        spark_functions.col("profit_taking_bar_index").isNull()
                        | (
                            spark_functions.col("stop_loss_bar_index")
                            < spark_functions.col("profit_taking_bar_index")
                        )
                    ),
                    spark_functions.lit("stop_loss"),
                )
                .when(
                    spark_functions.col("profit_taking_bar_index").isNotNull()
                    & (
                        spark_functions.col("profit_taking_bar_index")
                        == spark_functions.col("stop_loss_bar_index")
                    ),
                    spark_functions.lit("ambiguous"),
                )
                .otherwise(spark_functions.lit("timeout")),
            )
            .withColumn(
                "long_short_label",
                spark_functions.when(
                    spark_functions.col("barrier_outcome") == "profit_taking",
                    spark_functions.lit(1),
                )
                .when(
                    spark_functions.col("barrier_outcome") == "stop_loss",
                    spark_functions.lit(-1),
                )
                .otherwise(spark_functions.lit(0))
                .cast("int"),
            )
            .select(
                "timestamp",
                "symbol",
                spark_functions.col("long_short_label").alias("label_value"),
            )
        )
