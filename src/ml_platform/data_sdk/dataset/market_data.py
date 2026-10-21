"""Download hourly Yahoo Finance OHLCV bars into an Iceberg Parquet table."""

from datetime import date, datetime, time, timezone
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import os
from typing import Any
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType, TimestampType

from ml_platform.sdk.configs import MarketDataConfig, VariantsConfig, with_variants_config

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
OHLCV_SCHEMA = StructType(
    [
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("symbol", StringType(), nullable=False),
        StructField("Open", DoubleType(), nullable=False),
        StructField("High", DoubleType(), nullable=False),
        StructField("Low", DoubleType(), nullable=False),
        StructField("Close", DoubleType(), nullable=False),
        StructField("Volume", LongType(), nullable=False),
    ]
)
ICEBERG_VERSION = "1.12.0"
ICEBERG_CATALOG = "data"
ICEBERG_NAMESPACE = "market_data"
ICEBERG_TABLE = "hourly_ohlcv"
ICEBERG_TABLE_IDENTIFIER = f"{ICEBERG_CATALOG}.{ICEBERG_NAMESPACE}.{ICEBERG_TABLE}"
MARKET_DATASET_ID = f"{ICEBERG_NAMESPACE}.{ICEBERG_TABLE}"
DATA_ROOT = Path("data")


@dataclass(frozen=True)
class MaterializationResult:
    """Committed row count, Iceberg snapshot, and content hashes by requested timestamp."""

    row_count: int
    snapshot_id: int
    partition_hashes: dict[str, str]


def fetch_hourly_ohlcv(
    spark: SparkSession,
    symbol: str,
    *,
    start_date: date,
    end_date: date,
) -> DataFrame:
    """Fetch one symbol's extended-hours bars into a typed Spark DataFrame."""
    normalized_symbol = symbol.upper()
    rows = _yahoo_chart_rows(
        _request_yahoo_chart(normalized_symbol, start_date, end_date), normalized_symbol
    )
    if not rows:
        raise ValueError(f"Yahoo Finance returned no hourly data for {normalized_symbol}.")
    return _with_market_session(spark.createDataFrame(rows, schema=OHLCV_SCHEMA))


def materialize_daily_ohlcv_partitions(
    frame: DataFrame,
    warehouse_dir: str | Path,
    *,
    spark: SparkSession | None = None,
) -> int:
    """Merge a Spark OHLCV frame into a daily-partitioned Iceberg table."""
    required_columns = {"timestamp", "symbol", "market_session", *OHLCV_COLUMNS}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Cannot partition data without columns: {', '.join(sorted(missing_columns))}.")

    session = spark or frame.sparkSession
    incoming = frame.select(
        F.col("timestamp").cast("timestamp").alias("timestamp"),
        F.col("symbol").cast("string").alias("symbol"),
        F.col("market_session").cast("string").alias("market_session"),
        F.col("Open").cast("double").alias("Open"),
        F.col("High").cast("double").alias("High"),
        F.col("Low").cast("double").alias("Low"),
        F.col("Close").cast("double").alias("Close"),
        F.col("Volume").cast("long").alias("Volume"),
    )
    if incoming.where(F.col("timestamp").isNull()).limit(1).count():
        raise ValueError("Cannot partition rows with invalid timestamps.")
    if incoming.limit(1).count() == 0:
        raise ValueError("Cannot partition an empty OHLCV DataFrame.")

    partition_count = incoming.select(F.to_date("timestamp").alias("day")).distinct().count()
    _ensure_iceberg_table(session)
    (
        incoming.withColumn("_day", F.to_date("timestamp"))
        .repartition("_day")
        .sortWithinPartitions("_day", "timestamp", "symbol")
        .drop("_day")
        .createOrReplaceTempView("incoming_hourly_ohlcv")
    )
    session.sql(
        f"""
        MERGE INTO {ICEBERG_TABLE_IDENTIFIER} AS target
        USING incoming_hourly_ohlcv AS source
        ON target.symbol = source.symbol AND target.timestamp = source.timestamp
        WHEN MATCHED THEN UPDATE SET
            target.Open = source.Open, target.High = source.High, target.Low = source.Low,
            target.Close = source.Close, target.Volume = source.Volume,
            target.market_session = source.market_session
        WHEN NOT MATCHED THEN INSERT (
            timestamp, symbol, market_session, Open, High, Low, Close, Volume
        ) VALUES (
            source.timestamp, source.symbol, source.market_session, source.Open,
            source.High, source.Low, source.Close, source.Volume
        )
        """
    )
    return partition_count


def create_spark_session(warehouse_dir: str | Path = DATA_ROOT) -> SparkSession:
    """Create a local Spark session with an Iceberg Hadoop catalog in UTC."""
    warehouse = str(warehouse_dir) if "://" in str(warehouse_dir) else str(Path(warehouse_dir).resolve())
    builder = SparkSession.builder.appName("ml-platform-market-data")
    if os.environ.get("SPARK_MASTER"):
        builder = builder.master(os.environ["SPARK_MASTER"])
    return (
        builder
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.scheduler.mode", "FAIR")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config(f"spark.sql.catalog.{ICEBERG_CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{ICEBERG_CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{ICEBERG_CATALOG}.warehouse", warehouse)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def _request_yahoo_chart(symbol: str, start_date: date, end_date: date) -> dict[str, Any]:
    """Request one Yahoo chart response and return its decoded JSON payload."""
    period_start = int(datetime.combine(start_date, time.min, timezone.utc).timestamp())
    period_end = int(datetime.combine(end_date, time.min, timezone.utc).timestamp())
    url = (
        f"{YAHOO_CHART_URL}/{quote(symbol, safe='')}?period1={period_start}"
        f"&period2={period_end}&interval=1h&includePrePost=true&events=history"
    )
    with urlopen(Request(url, headers={"User-Agent": "ml-platform/0.1"}), timeout=30) as response:
        return json.load(response)


def _yahoo_chart_rows(payload: dict[str, Any], symbol: str) -> list[tuple[object, ...]]:
    """Validate Yahoo chart JSON and convert complete bars into typed row tuples."""
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError("Yahoo Finance response does not contain a chart payload.")
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance returned an error for {symbol}: {chart['error']}")
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return []
    result = results[0]
    if not isinstance(result, dict):
        raise ValueError("Yahoo Finance response contains an invalid chart result.")
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    quotes = indicators.get("quote") if isinstance(indicators, dict) else None
    values = quotes[0] if isinstance(quotes, list) and quotes else None
    if not isinstance(timestamps, list) or not isinstance(values, dict):
        raise ValueError("Yahoo Finance response is missing expected OHLCV columns.")
    fields = {column: values.get(column.lower()) for column in OHLCV_COLUMNS}
    missing = [column for column, column_values in fields.items() if not isinstance(column_values, list)]
    if missing:
        raise ValueError(f"Yahoo Finance response is missing expected columns: {', '.join(missing)}.")
    if any(len(column_values) != len(timestamps) for column_values in fields.values()):
        raise ValueError("Yahoo Finance response has inconsistent OHLCV column lengths.")

    rows: list[tuple[object, ...]] = []
    for timestamp, open_, high, low, close, volume in zip(
        timestamps, fields["Open"], fields["High"], fields["Low"], fields["Close"], fields["Volume"], strict=True
    ):
        if None in (timestamp, open_, high, low, close, volume):
            continue
        rows.append((datetime.fromtimestamp(timestamp, tz=timezone.utc), symbol, float(open_), float(high), float(low), float(close), int(volume)))
    return rows


def _with_market_session(frame: DataFrame) -> DataFrame:
    """Add a pre-market, regular, or after-hours label using Spark expressions."""
    eastern_time = F.date_format(F.from_utc_timestamp("timestamp", "America/New_York"), "HH:mm:ss")
    return frame.withColumn(
        "market_session",
        F.when((eastern_time >= "04:00:00") & (eastern_time < "09:30:00"), "pre_market")
        .when((eastern_time >= "09:30:00") & (eastern_time < "16:00:00"), "regular")
        .otherwise("after_hours"),
    ).select("timestamp", "symbol", "market_session", *OHLCV_COLUMNS)


def _ensure_iceberg_table(session: SparkSession) -> None:
    """Create the local Iceberg namespace and its Parquet-backed OHLCV table."""
    session.sql(f"CREATE NAMESPACE IF NOT EXISTS {ICEBERG_CATALOG}.{ICEBERG_NAMESPACE}")
    session.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {ICEBERG_TABLE_IDENTIFIER} (
            timestamp TIMESTAMP, symbol STRING, market_session STRING, Open DOUBLE,
            High DOUBLE, Low DOUBLE, Close DOUBLE, Volume BIGINT
        ) USING iceberg PARTITIONED BY (days(timestamp))
        TBLPROPERTIES (
            'format-version' = '2', 'write.format.default' = 'parquet',
            'write.parquet.compression-codec' = 'zstd',
            'write.distribution-mode' = 'hash', 'write.spark.fanout.enabled' = 'false'
        )
        """
    )


def materialize_market_data(config: MarketDataConfig, *, warehouse_dir: str | Path = DATA_ROOT) -> int:
    """Run the market-data Spark job and return its committed row count."""
    return materialize_market_data_with_receipts(config, warehouse_dir=warehouse_dir).row_count


def materialize_market_data_with_receipts(
    config: MarketDataConfig, *, warehouse_dir: str | Path = DATA_ROOT, requested_timestamps: list[str] | None = None
) -> MaterializationResult:
    """Run the market-data job and produce snapshot-backed partition receipts."""
    spark = create_spark_session(warehouse_dir)
    try:
        frames = []
        for symbol in config.symbols:
            try:
                frames.append(fetch_hourly_ohlcv(spark, symbol, start_date=config.time_range.start_date, end_date=config.time_range.end_date))
            except (HTTPError, ValueError) as error:
                if isinstance(error, HTTPError) and error.code not in {404, 422}:
                    raise
                print(f"Skipping unavailable Yahoo Finance symbol {symbol}: {error}")
        if not frames:
            raise RuntimeError("Yahoo Finance returned no data for any configured symbol.")
        frame = frames[0]
        for symbol_frame in frames[1:]:
            frame = frame.unionByName(symbol_frame)
        row_count = frame.count()
        materialize_daily_ohlcv_partitions(frame, warehouse_dir, spark=spark)
        snapshot = spark.sql(f"SELECT snapshot_id FROM {ICEBERG_TABLE_IDENTIFIER}.snapshots ORDER BY committed_at DESC LIMIT 1").first()
        if snapshot is None:
            raise RuntimeError("Iceberg materialization completed without a snapshot.")
        requested = requested_timestamps or []
        requested_days = {datetime.fromisoformat(timestamp).date().isoformat(): timestamp for timestamp in requested}
        partition_hashes: dict[str, str] = {}
        if requested_days:
            row_hash = F.sha2(F.to_json(F.struct("timestamp", "symbol", "market_session", "Open", "High", "Low", "Close", "Volume")), 256)
            rows = (spark.table(ICEBERG_TABLE_IDENTIFIER).withColumn("partition_day", F.date_format("timestamp", "yyyy-MM-dd")).where(F.col("partition_day").isin(sorted(requested_days))).groupBy("partition_day").agg(F.sha2(F.concat_ws("", F.sort_array(F.collect_list(row_hash))), 256).alias("content_hash")).collect())
            hashes = {row["partition_day"]: row["content_hash"] for row in rows}
            # Weekends and exchange holidays have no physical Iceberg file, but
            # are valid empty logical partitions in a requested calendar range.
            empty_partition_hash = sha256(b"").hexdigest()
            hashes = {day: hashes.get(day, empty_partition_hash) for day in requested_days}
            partition_hashes = {requested_days[day]: content_hash for day, content_hash in hashes.items()}
        return MaterializationResult(row_count=row_count, snapshot_id=int(snapshot["snapshot_id"]), partition_hashes=partition_hashes)
    finally:
        spark.stop()


@with_variants_config
def main(variants_config: VariantsConfig) -> None:
    """Run the market-data Spark job for the selected dataset configuration."""
    row_count = materialize_market_data(variants_config.datasets)
    print(f"Saved {row_count:,} hourly bars for {len(variants_config.datasets.symbols):,} symbols in {DATA_ROOT}")


if __name__ == "__main__":
    main()
