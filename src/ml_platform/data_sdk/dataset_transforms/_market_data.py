"""Yahoo Finance market-data dataset transform."""

from datetime import date, datetime, time, timedelta, timezone
import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType, TimestampType

from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.utils import record

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
OHLCV_SCHEMA = StructType([StructField("timestamp", TimestampType(), False), StructField("symbol", StringType(), False), StructField("Open", DoubleType(), False), StructField("High", DoubleType(), False), StructField("Low", DoubleType(), False), StructField("Close", DoubleType(), False), StructField("Volume", LongType(), False)])


@record
class MarketDataTransform(DatasetTransform):
    """Download configured Yahoo Finance bars and construct the OHLCV DataFrame."""

    spark: SparkSession
    symbols: list[str]
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
        """Fetch contiguous Yahoo ranges and return rows only for requested partition days."""
        requested_days = sorted({partition.date() for partition in self.partitions})
        if not requested_days:
            raise ValueError("Market-data materialization requires at least one partition.")
        frames: list[DataFrame] = []
        for symbol in self.symbols:
            for start_date, end_date in _contiguous_date_ranges(requested_days):
                try:
                    frames.append(fetch_hourly_ohlcv(self.spark, symbol, start_date=start_date, end_date=end_date))
                except (HTTPError, ValueError) as error:
                    if isinstance(error, HTTPError) and error.code not in {404, 422}:
                        raise
                    print(f"Skipping unavailable Yahoo Finance range for {symbol}: {error}")
        if not frames:
            raise RuntimeError("Yahoo Finance returned no data for any configured symbol.")
        result = frames[0]
        for frame in frames[1:]:
            result = result.unionByName(frame)
        return result.where(F.to_date("timestamp").isin(requested_days))


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


def fetch_hourly_ohlcv(spark: SparkSession, symbol: str, *, start_date: date, end_date: date) -> DataFrame:
    """Fetch one symbol's extended-hours bars into a typed Spark DataFrame."""
    normalized = symbol.upper()
    rows = _yahoo_chart_rows(_request_yahoo_chart(normalized, start_date, end_date), normalized)
    if not rows:
        raise ValueError(f"Yahoo Finance returned no hourly data for {normalized}.")
    return _with_market_session(spark.createDataFrame(rows, schema=OHLCV_SCHEMA))


def _request_yahoo_chart(symbol: str, start_date: date, end_date: date) -> dict[str, Any]:
    """Request one Yahoo chart response and return decoded JSON."""
    start = int(datetime.combine(start_date, time.min, timezone.utc).timestamp())
    end = int(datetime.combine(end_date, time.min, timezone.utc).timestamp())
    url = f"{YAHOO_CHART_URL}/{quote(symbol, safe='')}?period1={start}&period2={end}&interval=1h&includePrePost=true&events=history"
    with urlopen(Request(url, headers={"User-Agent": "ml-platform/0.1"}), timeout=30) as response:
        return json.load(response)


def _yahoo_chart_rows(payload: dict[str, Any], symbol: str) -> list[tuple[object, ...]]:
    """Validate Yahoo chart JSON and convert complete bars into typed row tuples."""
    chart = payload.get("chart")
    if not isinstance(chart, dict): raise ValueError("Yahoo Finance response does not contain a chart payload.")
    if chart.get("error"): raise ValueError(f"Yahoo Finance returned an error for {symbol}: {chart['error']}")
    results = chart.get("result")
    if not isinstance(results, list) or not results: return []
    result = results[0]
    if not isinstance(result, dict): raise ValueError("Yahoo Finance response contains an invalid chart result.")
    timestamps = result.get("timestamp"); indicators = result.get("indicators")
    quotes = indicators.get("quote") if isinstance(indicators, dict) else None
    values = quotes[0] if isinstance(quotes, list) and quotes else None
    if not isinstance(timestamps, list) or not isinstance(values, dict): raise ValueError("Yahoo Finance response is missing expected OHLCV columns.")
    fields = {column: values.get(column.lower()) for column in OHLCV_COLUMNS}
    missing = [column for column, values in fields.items() if not isinstance(values, list)]
    if missing: raise ValueError(f"Yahoo Finance response is missing expected columns: {', '.join(missing)}.")
    if any(len(values) != len(timestamps) for values in fields.values()): raise ValueError("Yahoo Finance response has inconsistent OHLCV column lengths.")
    return [(datetime.fromtimestamp(timestamp, tz=timezone.utc), symbol, float(open_), float(high), float(low), float(close), int(volume)) for timestamp, open_, high, low, close, volume in zip(timestamps, fields["Open"], fields["High"], fields["Low"], fields["Close"], fields["Volume"], strict=True) if None not in (timestamp, open_, high, low, close, volume)]


def _with_market_session(frame: DataFrame) -> DataFrame:
    """Add a pre-market, regular, or after-hours session label."""
    eastern = F.date_format(F.from_utc_timestamp("timestamp", "America/New_York"), "HH:mm:ss")
    return frame.withColumn("market_session", F.when((eastern >= "04:00:00") & (eastern < "09:30:00"), "pre_market").when((eastern >= "09:30:00") & (eastern < "16:00:00"), "regular").otherwise("after_hours")).select("timestamp", "symbol", "market_session", *OHLCV_COLUMNS)
