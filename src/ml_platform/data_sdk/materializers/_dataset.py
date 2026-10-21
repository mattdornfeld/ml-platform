"""API-dispatched Spark materializer for market-data datasets."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.request import Request, urlopen

from pydantic import Field, field_validator
from simple_classproperty import classproperty
from pyspark.sql import DataFrame, SparkSession, functions as F

from ml_platform.data_sdk.materializers._base import Materializer, MaterializerConfig, MaterializerPartition
from ml_platform.data_sdk.materializers._constants import materialization_api_url
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform


class DatasetMaterializationConfig(MaterializerConfig):
    """Complete JSON contract consumed by a dataset Spark materializer."""

    dataset_type: Literal["market_data"] = Field(description="Kind of dataset transform to execute.")
    symbols: list[str] = Field(min_length=1, description="Yahoo Finance ticker symbols to retrieve for each requested partition.")

    @field_validator("symbols")
    @classmethod
    def _normalize_symbols(cls, symbols: list[str]) -> list[str]:
        normalized = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not normalized:
            raise ValueError("Provide at least one non-empty market symbol.")
        return normalized


class DatasetMaterializer(Materializer[DatasetMaterializationConfig]):
    """Construct, validate, persist, and report one market-data materialization job."""

    config_model = DatasetMaterializationConfig
    spark_application_name: ClassVar[str] = "ml-platform-dataset-materializer"
    data_root: ClassVar[Path] = Path("data")
    ohlcv_columns: ClassVar[tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")
    iceberg_schema: ClassVar[str] = (
        "timestamp TIMESTAMP, symbol STRING, market_session STRING, Open DOUBLE, "
        "High DOUBLE, Low DOUBLE, Close DOUBLE, Volume BIGINT"
    )
    iceberg_partition_spec: ClassVar[str] = "days(timestamp)"

    @classproperty
    def iceberg_namespace(cls) -> str:
        """Store market-data resources in their dedicated Iceberg namespace."""
        return "market_data"

    @classmethod
    def _materialize_frame(
        cls, frame: DataFrame, *, spark: SparkSession, name: str, version: int
    ) -> int:
        """Validate and merge one transform's OHLCV frame into its versioned table."""
        cls._validate_required_columns(
            frame, {"timestamp", "symbol", "market_session", *cls.ohlcv_columns}, context="Dataset transform"
        )
        incoming = frame.select(
            F.col("timestamp").cast("timestamp").alias("timestamp"),
            F.col("symbol").cast("string").alias("symbol"),
            F.col("market_session").cast("string").alias("market_session"),
            *[F.col(column).cast("long" if column == "Volume" else "double").alias(column) for column in cls.ohlcv_columns],
        )
        if incoming.where(F.col("timestamp").isNull() | F.col("symbol").isNull()).limit(1).count():
            raise ValueError("Dataset transform returned invalid timestamp or symbol values.")
        cls._validate_non_empty(incoming, context="Dataset transform")
        cls._ensure_iceberg_table(spark, name=name, version=version)
        identifier = cls._iceberg_table_identifier(name, version)
        (
            incoming.withColumn("_day", F.to_date("timestamp"))
            .repartition("_day")
            .sortWithinPartitions("_day", "timestamp", "symbol")
            .drop("_day")
            .createOrReplaceTempView("incoming_hourly_ohlcv")
        )
        spark.sql(
            f"""MERGE INTO {identifier} AS target
            USING incoming_hourly_ohlcv AS source
            ON target.symbol = source.symbol AND target.timestamp = source.timestamp
            WHEN MATCHED THEN UPDATE SET
                target.Open = source.Open, target.High = source.High, target.Low = source.Low,
                target.Close = source.Close, target.Volume = source.Volume,
                target.market_session = source.market_session
            WHEN NOT MATCHED THEN INSERT (timestamp, symbol, market_session, Open, High, Low, Close, Volume)
            VALUES (source.timestamp, source.symbol, source.market_session, source.Open, source.High,
                source.Low, source.Close, source.Volume)"""
        )
        return incoming.count()

    @classmethod
    def _materialize_test_frame(
        cls, frame: DataFrame, *, name: str, version: int, spark: SparkSession | None = None
    ) -> int:
        """Persist a prepared OHLCV frame for local integration callers."""
        return cls._materialize_frame(frame, spark=spark or frame.sparkSession, name=name, version=version)

    @classmethod
    def _complete_partitions(
        cls,
        spark: SparkSession,
        partitions: list[MaterializerPartition],
        snapshot_id: int,
        table_identifier: str,
    ) -> None:
        """Compute logical daily hashes and report completed API targets."""
        requested = {item.timestamp.date().isoformat(): item for item in partitions}
        digest = F.sha2(
            F.to_json(F.struct("timestamp", "symbol", "market_session", "Open", "High", "Low", "Close", "Volume")), 256
        )
        rows = (
            spark.table(table_identifier)
            .withColumn("day", F.date_format("timestamp", "yyyy-MM-dd"))
            .where(F.col("day").isin(sorted(requested)))
            .groupBy("day")
            .agg(F.sha2(F.concat_ws("", F.sort_array(F.collect_list(digest))), 256).alias("content_hash"))
            .collect()
        )
        hashes = {row["day"]: row["content_hash"] for row in rows}
        empty_hash = sha256(b"").hexdigest()
        for day, partition in requested.items():
            body = json.dumps({"content_hash": hashes.get(day, empty_hash), "snapshot_id": snapshot_id}).encode()
            request = Request(
                f"{materialization_api_url()}"
                f"/dataset-partition-update-requests/{partition.partition_id}/complete",
                data=body,
                headers={"Content-Type": "application/json", "callback-token": partition.callback_token},
                method="POST",
            )
            with urlopen(request, timeout=30):
                pass

    def materialize(self, spark: SparkSession) -> int:
        """Construct the dataset transform, persist its rows, and complete partitions."""
        from ml_platform.data_sdk.dataset_transforms._market_data import MarketDataTransform

        transform: DatasetTransform = MarketDataTransform(
            spark=spark,
            symbols=self.config.symbols,
            partitions=[partition.timestamp for partition in self.config.partitions],
        )
        row_count = self._materialize_frame(
            transform(), spark=spark, name=self.config.name, version=self.config.version
        )
        snapshot_id = self._latest_snapshot_id(spark, self._table_identifier)
        self._complete_partitions(spark, self.config.partitions, snapshot_id, self._table_identifier)
        return row_count


if __name__ == "__main__":
    DatasetMaterializer.main()
