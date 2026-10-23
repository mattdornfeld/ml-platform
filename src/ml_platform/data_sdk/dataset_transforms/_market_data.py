"""Yahoo Finance market-data dataset transform."""

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as spark_functions
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from ml_platform.common.types import Symbol
from ml_platform.data_sdk._exceptions import DataSdkError
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.utils import record

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


@dataclass(frozen=True)
class HourlyOhlcvRow:
    """One complete Yahoo Finance hourly OHLCV observation."""

    timestamp: datetime
    symbol: Symbol
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int


class MarketDataPartitionError(DataSdkError):
    """Raised when a market-data transform is invoked without output partitions."""


class YahooFinanceDataUnavailableError(DataSdkError):
    """Raised when Yahoo Finance returns no hourly bars for a requested symbol or job."""


class YahooFinancePayloadError(DataSdkError):
    """Raised when Yahoo Finance returns a malformed or error chart payload."""


@record
class MarketDataTransform(DatasetTransform):
    """Download configured Yahoo Finance bars and construct the OHLCV DataFrame."""

    spark: SparkSession
    symbols: Sequence[Symbol]
    partitions: list[datetime]

    @property
    def value_schema(self) -> dict[str, str]:
        """Declare the OHLCV columns persisted beside the shared transform keys."""
        return {
            "market_session": "string",
            "Open": "double",
            "High": "double",
            "Low": "double",
            "Close": "double",
            "Volume": "bigint",
        }

    def __call__(self) -> DataFrame:
        """Fetch contiguous Yahoo ranges for the requested partition days."""
        requested_days = sorted(
            {partition.date() for partition in self.partitions}
        )
        if not requested_days:
            raise MarketDataPartitionError(
                "Market-data materialization requires at least one requested output partition."
            )
        rows: list[HourlyOhlcvRow] = []
        for symbol in self.symbols:
            for start_date, end_date in _contiguous_date_ranges(requested_days):
                try:
                    rows.extend(
                        fetch_hourly_ohlcv_rows(
                            symbol,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    )
                except (HTTPError, ValueError) as error:
                    if isinstance(error, HTTPError) and error.code not in {
                        404,
                        422,
                    }:
                        raise
                    sys.stdout.write(
                        f"Skipping unavailable Yahoo Finance range for {symbol}: {error}\n"
                    )
        if not rows:
            raise YahooFinanceDataUnavailableError(
                "Yahoo Finance returned no hourly data for any configured symbol "
                f"({', '.join(self.symbols)})."
            )
        return _with_market_session(_market_data_frame(self.spark, rows))


def _contiguous_date_ranges(partitions: list[date]) -> list[tuple[date, date]]:
    """Collapse requested calendar days into Yahoo-friendly half-open request ranges."""
    if not partitions:
        return []
    ranges: list[tuple[date, date]] = []
    start = previous = partitions[0]
    for current in partitions[1:]:
        if current != previous + timedelta(days=1):
            ranges.append((start, previous + timedelta(days=1)))
            start = current
        previous = current
    ranges.append((start, previous + timedelta(days=1)))
    return ranges


def fetch_hourly_ohlcv(
    spark: SparkSession, symbol: Symbol, *, start_date: date, end_date: date
) -> DataFrame:
    """Fetch one symbol's extended-hours bars into a typed Spark DataFrame."""
    rows = fetch_hourly_ohlcv_rows(
        symbol,
        start_date=start_date,
        end_date=end_date,
    )
    return _with_market_session(_market_data_frame(spark, rows))


def fetch_hourly_ohlcv_rows(
    symbol: Symbol, *, start_date: date, end_date: date
) -> list[HourlyOhlcvRow]:
    """Fetch one symbol's complete extended-hours OHLCV rows from Yahoo Finance."""
    normalized = Symbol(symbol.upper())
    rows = _yahoo_chart_rows(
        _request_yahoo_chart(normalized, start_date, end_date), normalized
    )
    if not rows:
        raise YahooFinanceDataUnavailableError(
            f"Yahoo Finance returned no hourly data for symbol {normalized!r} "
            f"from {start_date.isoformat()} through {end_date.isoformat()}."
        )
    return rows


def _request_yahoo_chart(
    symbol: Symbol, start_date: date, end_date: date
) -> dict[str, Any]:
    """Request one Yahoo chart response and return decoded JSON."""
    start = int(datetime.combine(start_date, time.min, UTC).timestamp())
    end = int(datetime.combine(end_date, time.min, UTC).timestamp())
    url = f"{YAHOO_CHART_URL}/{quote(symbol, safe='')}?period1={start}&period2={end}&interval=1h&includePrePost=true&events=history"
    with urlopen(
        Request(url, headers={"User-Agent": "ml-platform/0.1"}), timeout=30
    ) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise YahooFinancePayloadError(
            f"Yahoo Finance response for {symbol!r} is not a JSON object."
        )
    return {str(key): value for key, value in payload.items()}


def _yahoo_chart_rows(
    payload: dict[str, Any], symbol: str
) -> list[HourlyOhlcvRow]:
    """Validate Yahoo chart JSON and convert complete bars into typed OHLCV rows."""
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise YahooFinancePayloadError(
            f"Yahoo Finance response for {symbol!r} does not contain a chart payload."
        )
    if chart.get("error"):
        raise YahooFinancePayloadError(
            f"Yahoo Finance returned an error for {symbol!r}: {chart['error']}."
        )
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return []
    result = results[0]
    if not isinstance(result, dict):
        raise YahooFinancePayloadError(
            f"Yahoo Finance response for {symbol!r} contains an invalid chart result."
        )
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    quotes = indicators.get("quote") if isinstance(indicators, dict) else None
    values = quotes[0] if isinstance(quotes, list) and quotes else None
    if not isinstance(timestamps, list) or not isinstance(values, dict):
        raise YahooFinancePayloadError(
            f"Yahoo Finance response for {symbol!r} is missing timestamp or OHLCV data."
        )
    open_values = values.get("open")
    high_values = values.get("high")
    low_values = values.get("low")
    close_values = values.get("close")
    volume_values = values.get("volume")
    if (
        not isinstance(open_values, list)
        or not isinstance(high_values, list)
        or not isinstance(low_values, list)
        or not isinstance(close_values, list)
        or not isinstance(volume_values, list)
    ):
        raise YahooFinancePayloadError(
            f"Yahoo Finance response for {symbol!r} is missing one or more OHLCV columns."
        )
    if any(
        len(column_values) != len(timestamps)
        for column_values in (
            open_values,
            high_values,
            low_values,
            close_values,
            volume_values,
        )
    ):
        raise YahooFinancePayloadError(
            f"Yahoo Finance response for {symbol!r} has OHLCV arrays inconsistent with timestamp count "
            f"{len(timestamps)}."
        )
    return [
        HourlyOhlcvRow(
            timestamp=datetime.fromtimestamp(timestamp, tz=UTC),
            symbol=Symbol(symbol),
            open_price=float(open_),
            high_price=float(high),
            low_price=float(low),
            close_price=float(close),
            volume=int(volume),
        )
        for timestamp, open_, high, low, close, volume in zip(
            timestamps,
            open_values,
            high_values,
            low_values,
            close_values,
            volume_values,
            strict=True,
        )
        if None not in (timestamp, open_, high, low, close, volume)
    ]


def _market_data_frame(
    spark: SparkSession, rows: list[HourlyOhlcvRow]
) -> DataFrame:
    """Convert explicitly typed OHLCV rows to the fixed Spark schema."""
    return spark.createDataFrame(
        [
            {
                "timestamp": row.timestamp,
                "symbol": row.symbol,
                "Open": row.open_price,
                "High": row.high_price,
                "Low": row.low_price,
                "Close": row.close_price,
                "Volume": row.volume,
            }
            for row in rows
        ],
        schema=OHLCV_SCHEMA,
    )


def _with_market_session(frame: DataFrame) -> DataFrame:
    """Add a pre-market, regular, or after-hours session label."""
    eastern = spark_functions.date_format(
        spark_functions.from_utc_timestamp("timestamp", "America/New_York"),
        "HH:mm:ss",
    )
    return frame.withColumn(
        "market_session",
        spark_functions.when(
            (eastern >= "04:00:00") & (eastern < "09:30:00"), "pre_market"
        )
        .when((eastern >= "09:30:00") & (eastern < "16:00:00"), "regular")
        .otherwise("after_hours"),
    ).select("timestamp", "symbol", "market_session", *OHLCV_COLUMNS)
