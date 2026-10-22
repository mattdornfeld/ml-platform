"""API-dispatched Spark materializer for market-data datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import Field, field_validator
from simple_classproperty import classproperty
from pyspark.sql import DataFrame, SparkSession, functions as F

from ml_platform.data_sdk.materializers._base import Materializer, MaterializerConfig
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.transforms._types import FeatureKey, PartitionType


class DatasetMaterializationConfig(MaterializerConfig):
    """Complete JSON contract consumed by a dataset Spark materializer."""

    dataset_type: str = Field(min_length=1, description="Kind of dataset transform to execute.")
    keys: list[FeatureKey] = Field(
        min_length=1,
        description="Ordered logical entity keys emitted by the dataset transform.",
    )
    partition_type: PartitionType = Field(
        description="Iceberg timestamp partition cadence used for this dataset.",
    )
    symbols: list[str] = Field(min_length=1, description="Yahoo Finance ticker symbols to retrieve for each requested partition.")

    @field_validator("symbols")
    @classmethod
    def _normalize_symbols(cls, symbols: list[str]) -> list[str]:
        normalized = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not normalized:
            raise ValueError("Provide at least one non-empty market symbol.")
        return normalized


@dataclass(frozen=True)
class DatasetMaterializer(Materializer[DatasetMaterializationConfig]):
    """Construct, validate, persist, and report one market-data materialization job."""

    config_model = DatasetMaterializationConfig
    spark_application_name: ClassVar[str] = "ml-platform-dataset-materializer"
    data_root: ClassVar[Path] = Path("data")
    ohlcv_columns: ClassVar[tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")
    iceberg_merge_keys: ClassVar[tuple[str, ...]] = ("symbol", "timestamp")
    callback_resource: ClassVar[str] = "dataset"
    allow_empty_partitions: ClassVar[bool] = True
    content_hash_columns: ClassVar[tuple[str, ...]] = (
        "timestamp", "symbol", "market_session", "Open", "High", "Low", "Close", "Volume"
    )

    @classproperty
    def iceberg_namespace(cls) -> str:
        """Store market-data resources in their dedicated Iceberg namespace."""
        return "market_data"

    @classmethod
    def _materialize_frame(
        cls, frame: DataFrame, *, spark: SparkSession, config: DatasetMaterializationConfig
    ) -> int:
        """Validate and merge one transform's OHLCV frame into its versioned table."""
        materializer = cls(config)
        cls._validate_required_columns(
            frame, {"timestamp", "symbol", "market_session", *cls.ohlcv_columns}, context="Dataset transform"
        )
        incoming = frame.select(
            F.col("timestamp").cast("timestamp").alias("timestamp"),
            F.col("symbol").cast("string").alias("symbol"),
            F.col("market_session").cast("string").alias("market_session"),
            *[F.col(column).cast("long" if column == "Volume" else "double").alias(column) for column in cls.ohlcv_columns],
        )
        cls._validate_non_null_columns(incoming, ("timestamp", "symbol"), context="Dataset transform")
        cls._validate_non_empty(incoming, context="Dataset transform")
        transform = materializer._create_transform(spark)
        materializer._ensure_table(
            spark,
            schema=transform.iceberg_schema,
            partition_spec=transform.iceberg_partition_spec,
        )
        identifier = materializer._table_identifier
        source = (
            incoming.withColumn("_day", F.to_date("timestamp"))
            .repartition("_day")
            .sortWithinPartitions("_day", "timestamp", "symbol")
            .drop("_day")
        )
        cls._merge_frame(source, identifier, cls.iceberg_merge_keys)
        return incoming.count()

    @classmethod
    def _materialize_test_frame(
        cls, frame: DataFrame, *, config: DatasetMaterializationConfig, spark: SparkSession | None = None
    ) -> int:
        """Persist a prepared OHLCV frame for local integration callers."""
        return cls._materialize_frame(frame, spark=spark or frame.sparkSession, config=config)

    def _prepare_frame(self, frame: DataFrame) -> DataFrame:
        """Cast OHLCV values to the dataset's persisted types."""
        self._validate_required_columns(
            frame, set(self.content_hash_columns), context="Dataset transform"
        )
        return frame.select(
            F.col("timestamp").try_cast("timestamp").alias("timestamp"),
            F.col("symbol").cast("string").alias("symbol"),
            F.col("market_session").cast("string").alias("market_session"),
            *[
                F.col(column).try_cast("long" if column == "Volume" else "double").alias(column)
                for column in self.ohlcv_columns
            ],
        )

    def _validate_frame(self, frame: DataFrame, spark: SparkSession, partition_spec: str) -> None:
        """Require complete, unique market rows in only the requested days."""
        self._validate_non_null_columns(frame, self.content_hash_columns, context="Dataset transform")
        self._validate_unique_key(frame, ("timestamp", "symbol"), context="Dataset transform")
        requested_days = sorted({partition.timestamp.date() for partition in self.config.partitions})
        unexpected_rows = frame.where(~F.to_date("timestamp").isin(requested_days))
        if unexpected_rows.limit(1).count():
            raise ValueError("Dataset transform emitted rows outside the requested partitions.")

    def _write_frame(self, frame: DataFrame) -> None:
        """Cluster source rows by day before the base class merges them."""
        source = (
            frame.withColumn("_day", F.to_date("timestamp"))
            .repartition("_day")
            .sortWithinPartitions("_day", "timestamp", "symbol")
            .drop("_day")
        )
        super()._write_frame(source)

    def _create_transform(self, spark: SparkSession) -> DatasetTransform:
        """Build the user-facing transform selected by the dataset definition."""
        match self.config.dataset_type:
            case "market_data":
                from ml_platform.data_sdk.dataset_transforms._market_data import MarketDataTransform

                return MarketDataTransform(
                    spark=spark,
                    symbols=self.config.symbols,
                    partitions=[partition.timestamp for partition in self.config.partitions],
                    keys=self.config.keys,
                    partition_type=self.config.partition_type,
                )
            case unsupported_type:
                raise NotImplementedError(f"Unsupported dataset type: {unsupported_type}.")

    def materialize(self, spark: SparkSession, transform: DatasetTransform) -> DataFrame:
        """Construct the dataset DataFrame for the requested partitions."""
        return transform()


if __name__ == "__main__":
    DatasetMaterializer.main()
