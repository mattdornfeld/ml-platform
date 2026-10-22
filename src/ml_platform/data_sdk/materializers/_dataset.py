"""API-dispatched Spark materializer for market-data datasets."""
from dataclasses import dataclass

from pydantic import Field, field_validator
from simple_classproperty import classproperty
from pyspark.sql import DataFrame, SparkSession, functions as F

from ml_platform.data_sdk.materializers._base import Materializer, MaterializerConfig, WriteMode
from ml_platform.data_sdk.materializers._materialization_validator import DatasetValidationInput, MaterializationValidator
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.transforms._types import FeatureKey, PartitionType

_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_CONTENT_HASH_COLUMNS = ("timestamp", "symbol", "market_session", *_OHLCV_COLUMNS)
_MERGE_KEYS = ("symbol", "timestamp")


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

    @classproperty
    def config_model(cls) -> type[DatasetMaterializationConfig]:
        """Return the strict runtime contract for market-data jobs."""
        return DatasetMaterializationConfig

    @classproperty
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for market-data jobs."""
        return "ml-platform-dataset-materializer"

    @classproperty
    def iceberg_namespace(cls) -> str:
        """Store market-data resources in their dedicated Iceberg namespace."""
        return "market_data"

    @classproperty
    def callback_resource(cls) -> str:
        """Report market-data partition receipts as dataset resources."""
        return "dataset"

    @classproperty
    def write_mode(cls) -> WriteMode:
        """Merge market rows by their stable timestamp and symbol identity."""
        return WriteMode.MERGE

    @classmethod
    def _materialize_frame(
        cls, frame: DataFrame, *, spark: SparkSession, config: DatasetMaterializationConfig
    ) -> int:
        """Validate and merge one transform's OHLCV frame into its versioned table."""
        materializer = cls(config)
        incoming = materializer._prepare_frame(frame)
        transform = materializer._create_transform(spark)
        materializer.validate(incoming, spark, transform)
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
        cls._merge_frame(source, identifier, _MERGE_KEYS)
        return incoming.count()

    @classmethod
    def _materialize_test_frame(
        cls, frame: DataFrame, *, config: DatasetMaterializationConfig, spark: SparkSession | None = None
    ) -> int:
        """Persist a prepared OHLCV frame for local integration callers."""
        return cls._materialize_frame(frame, spark=spark or frame.sparkSession, config=config)

    def _prepare_frame(self, frame: DataFrame) -> DataFrame:
        """Cast OHLCV values to the dataset's persisted types."""
        return frame.select(
            F.col("timestamp").try_cast("timestamp").alias("timestamp"),
            F.col("symbol").cast("string").alias("symbol"),
            F.col("market_session").cast("string").alias("market_session"),
            *[
                F.col(column).try_cast("long" if column == "Volume" else "double").alias(column)
                for column in _OHLCV_COLUMNS
            ],
        )

    def _write_frame(self, frame: DataFrame) -> None:
        """Cluster source rows by day before the base class merges them."""
        source = (
            frame.withColumn("_day", F.to_date("timestamp"))
            .repartition("_day")
            .sortWithinPartitions("_day", "timestamp", "symbol")
            .drop("_day")
        )
        self._merge_frame(source, self._table_identifier, _MERGE_KEYS)

    def _row_digest(self) -> F.Column:
        """Hash every persisted market-data value for partition completion receipts."""
        return F.sha2(F.to_json(F.struct(*(F.col(column) for column in _CONTENT_HASH_COLUMNS))), 256)

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

    def validate(self, frame: DataFrame, spark: SparkSession, transform: DatasetTransform) -> None:
        """Require complete, unique market rows in only the requested partitions."""
        partition_spec = transform.iceberg_partition_spec
        requested_partitions = spark.createDataFrame(
            [(self._partition_timestamp(partition.timestamp, partition_spec),) for partition in self.config.partitions],
            "output_partition timestamp",
        ).distinct()
        partitioned = frame.withColumn(
            "output_partition", F.date_trunc(self._partition_unit(partition_spec), F.col("timestamp"))
        )
        self._validator.validate(
            DatasetValidationInput(
                frame=frame,
                partitioned_frame=partitioned,
                requested_partitions=requested_partitions,
                partition_column="output_partition",
                partition_interval=self._partition_interval(partition_spec),
                exact_columns=frozenset(_CONTENT_HASH_COLUMNS),
                non_null_columns=_CONTENT_HASH_COLUMNS,
                unique_key_columns=_MERGE_KEYS,
            )
        )


    def materialize(self, spark: SparkSession, transform: DatasetTransform) -> DataFrame:
        """Construct the dataset DataFrame for the requested partitions."""
        return transform()


if __name__ == "__main__":
    DatasetMaterializer.main()
