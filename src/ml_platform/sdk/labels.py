"""Create volatility-scaled triple-barrier labels from Iceberg OHLCV data."""

from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F

from ml_platform.sdk.configs import (
    LongShortLabelConfig,
    VariantsConfig,
    with_variants_config,
)
from ml_platform.data_sdk.dataset.market_data import (
    DATA_ROOT,
    ICEBERG_CATALOG,
    ICEBERG_TABLE_IDENTIFIER,
    create_spark_session,
)

LABEL_NAMESPACE = "labels"


def label_id(config: LongShortLabelConfig) -> str:
    """Return the versioned identifier used for one label dataset."""
    return f"{config.name}_v{config.version}"


def label_table_identifier(config: LongShortLabelConfig) -> str:
    """Return the Iceberg table identifier for one label dataset."""
    return f"{ICEBERG_CATALOG}.{LABEL_NAMESPACE}.{label_id(config)}"


def label_storage_path(warehouse_dir: str | Path, config: LongShortLabelConfig) -> Path:
    """Return the versioned label directory under the configured data root."""
    return Path(warehouse_dir).resolve() / LABEL_NAMESPACE / label_id(config)


def materialize_long_short_labels(
    warehouse_dir: str | Path,
    config: LongShortLabelConfig,
    *,
    spark: SparkSession | None = None,
) -> int:
    """Write triple-barrier labels from hourly OHLCV data to an Iceberg table.

    Each event uses volatility calculated from only preceding log returns. Its
    upper and lower barriers are the event close plus or minus the configured
    volatility multiples. Spark searches the next ``horizon_bars`` observations
    and records ``1`` for profit-taking, ``-1`` for stop-loss, and ``0`` for a
    time-out or same-bar dual barrier hit. Events without a complete vertical
    horizon are omitted.
    """
    session = spark or create_spark_session(warehouse_dir)
    if not session.catalog.tableExists(ICEBERG_TABLE_IDENTIFIER):
        raise ValueError(
            f"Market-data table does not exist: {ICEBERG_TABLE_IDENTIFIER}. "
            "Run ml-platform-download-ohlcv first."
        )

    by_symbol = Window.partitionBy("symbol").orderBy("timestamp")
    volatility_window = by_symbol.rowsBetween(-config.volatility_lookback_bars, -1)
    source = session.table(ICEBERG_TABLE_IDENTIFIER).select(
        "timestamp", "symbol", "High", "Low", "Close"
    )
    indexed_observations = (
        source.withColumn("bar_index", F.row_number().over(by_symbol))
    )
    observations = (
        indexed_observations
        .withColumn("previous_close", F.lag("Close").over(by_symbol))
        .withColumn("log_return", F.log(F.col("Close") / F.col("previous_close")))
        .withColumn("volatility", F.stddev_samp("log_return").over(volatility_window))
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
                ).over(by_symbol.rowsBetween(1, config.horizon_bars))
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
            F.col("close") * (1 + config.profit_taking_multiplier * F.col("volatility")),
        )
        .withColumn(
            "lower_barrier",
            F.col("close") * (1 - config.stop_loss_multiplier * F.col("volatility")),
        )
    )
    profit_hits = F.filter(
        F.col("future_bars"), lambda bar: bar["future_high"] >= F.col("upper_barrier")
    )
    stop_hits = F.filter(
        F.col("future_bars"), lambda bar: bar["future_low"] <= F.col("lower_barrier")
    )
    labels = (
        events.where(F.size("future_bars") == config.horizon_bars)
        .withColumn("profit_hit", F.try_element_at(profit_hits, F.lit(1)))
        .withColumn("stop_hit", F.try_element_at(stop_hits, F.lit(1)))
        .withColumn(
            "vertical_barrier_timestamp",
            F.element_at("future_bars", -1).getField("future_timestamp"),
        )
        .withColumn("profit_taking_bar_index", F.col("profit_hit").getField("future_bar_index"))
        .withColumn("stop_loss_bar_index", F.col("stop_hit").getField("future_bar_index"))
        .withColumn("timestamp", F.col("event_timestamp"))
        .withColumn("symbol", F.col("event_symbol"))
        .withColumn("horizon_bars", F.lit(config.horizon_bars).cast("int"))
        .withColumn(
            "barrier_outcome",
            F.when(
                F.col("profit_taking_bar_index").isNotNull()
                & (
                    F.col("stop_loss_bar_index").isNull()
                    | (F.col("profit_taking_bar_index") < F.col("stop_loss_bar_index"))
                ),
                F.lit("profit_taking"),
            )
            .when(
                F.col("stop_loss_bar_index").isNotNull()
                & (
                    F.col("profit_taking_bar_index").isNull()
                    | (F.col("stop_loss_bar_index") < F.col("profit_taking_bar_index"))
                ),
                F.lit("stop_loss"),
            )
            .when(
                F.col("profit_taking_bar_index").isNotNull()
                & (F.col("profit_taking_bar_index") == F.col("stop_loss_bar_index")),
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
            "vertical_barrier_timestamp",
            "close",
            "volatility",
            "upper_barrier",
            "lower_barrier",
            "horizon_bars",
            "barrier_outcome",
            "long_short_label",
        )
    )
    materialized_labels = labels.localCheckpoint(eager=True)
    row_count = materialized_labels.count()
    label_table = label_table_identifier(config)
    _ensure_label_table(session, config)
    materialized_labels.createOrReplaceTempView("incoming_long_short_labels")
    session.sql(
        f"""
        MERGE INTO {label_table} AS target
        USING incoming_long_short_labels AS source
        ON target.symbol = source.symbol
           AND target.timestamp = source.timestamp
           AND target.horizon_bars = source.horizon_bars
        WHEN MATCHED THEN UPDATE SET
            target.vertical_barrier_timestamp = source.vertical_barrier_timestamp,
            target.close = source.close,
            target.volatility = source.volatility,
            target.upper_barrier = source.upper_barrier,
            target.lower_barrier = source.lower_barrier,
            target.barrier_outcome = source.barrier_outcome,
            target.long_short_label = source.long_short_label
        WHEN NOT MATCHED THEN INSERT (
            timestamp, symbol, vertical_barrier_timestamp, close, volatility,
            upper_barrier, lower_barrier, horizon_bars, barrier_outcome, long_short_label
        ) VALUES (
            source.timestamp, source.symbol, source.vertical_barrier_timestamp, source.close,
            source.volatility, source.upper_barrier, source.lower_barrier,
            source.horizon_bars, source.barrier_outcome, source.long_short_label
        )
        """
    )
    return row_count


def _ensure_label_table(session: SparkSession, config: LongShortLabelConfig) -> None:
    """Create the daily-partitioned Iceberg triple-barrier label table when needed."""
    session.sql(f"CREATE NAMESPACE IF NOT EXISTS {ICEBERG_CATALOG}.{LABEL_NAMESPACE}")
    session.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {label_table_identifier(config)} (
            timestamp TIMESTAMP,
            symbol STRING,
            vertical_barrier_timestamp TIMESTAMP,
            close DOUBLE,
            volatility DOUBLE,
            upper_barrier DOUBLE,
            lower_barrier DOUBLE,
            horizon_bars INT,
            barrier_outcome STRING,
            long_short_label INT
        )
        USING iceberg
        PARTITIONED BY (days(timestamp))
        TBLPROPERTIES (
            'format-version' = '2',
            'write.format.default' = 'parquet',
            'write.parquet.compression-codec' = 'zstd',
            'write.distribution-mode' = 'hash',
            'write.spark.fanout.enabled' = 'false',
            'label.name' = '{config.name}',
            'label.version' = '{config.version}'
        )
        """
    )


@with_variants_config
def main(variants_config: VariantsConfig) -> None:
    """Extract triple-barrier labels using the selected variant's settings."""
    row_count = materialize_long_short_labels(DATA_ROOT, variants_config.labels)
    print(f"Saved {row_count:,} triple-barrier labels in {DATA_ROOT}")


if __name__ == "__main__":
    main()
