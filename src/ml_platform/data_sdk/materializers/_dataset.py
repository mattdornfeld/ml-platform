"""API-dispatched Spark materializer for market-data datasets."""
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

from pydantic import Field, field_validator
from pyspark.sql import DataFrame, SparkSession, functions as F

from ml_platform.data_sdk.materializers._base import Materializer, MaterializerConfig, WriteMode
from ml_platform.data_sdk.materializers._materialization_validator import (
    ExactColumnsValidationConfig,
    FiniteNumericValuesValidationConfig,
    NonNullValuesValidationConfig,
    UniqueKeyValidationConfig,
    ValidationConfig,
    XnysRegularSessionTemporalFilterConfig,
)
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

    @cached_property
    def partition_timestamps(self) -> tuple[datetime, ...]:
        """Return requested timestamps normalized to this dataset's cadence."""
        return self._partition_timestamps_for(self.partition_type)


@dataclass(frozen=True)
class DatasetMaterializer(Materializer[DatasetMaterializationConfig]):
    """Construct, validate, persist, and report one market-data materialization job."""

    @classmethod
    def config_model(cls) -> type[DatasetMaterializationConfig]:
        """Return the strict runtime contract for market-data jobs."""
        return DatasetMaterializationConfig

    @classmethod
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for market-data jobs."""
        return "ml-platform-dataset-materializer"

    @classmethod
    def iceberg_namespace(cls) -> str:
        """Store market-data resources in their dedicated Iceberg namespace."""
        return "market_data"

    @classmethod
    def callback_resource(cls) -> str:
        """Report market-data partition receipts as dataset resources."""
        return "dataset"

    @classmethod
    def write_mode(cls) -> WriteMode:
        """Merge market rows by their stable timestamp and symbol identity."""
        return WriteMode.MERGE

    @classmethod
    def _materialize_frame(
        cls, data_frame: DataFrame, *, spark: SparkSession, config: DatasetMaterializationConfig
    ) -> int:
        """Validate and merge one transform's OHLCV data_frame into its versioned table."""
        materializer = cls(config)
        incoming = materializer._prepare_frame(data_frame)
        transform = materializer._create_transform(spark)
        materializer.validate(incoming)
        materializer._create_table(
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
        cls, data_frame: DataFrame, *, config: DatasetMaterializationConfig, spark: SparkSession | None = None
    ) -> int:
        """Persist a prepared OHLCV data_frame for local integration callers."""
        return cls._materialize_frame(data_frame, spark=spark or data_frame.sparkSession, config=config)

    def _prepare_frame(self, data_frame: DataFrame) -> DataFrame:
        """Cast OHLCV values to the dataset's persisted types."""
        return data_frame.select(
            F.col("timestamp").try_cast("timestamp").alias("timestamp"),
            F.col("symbol").cast("string").alias("symbol"),
            F.col("market_session").cast("string").alias("market_session"),
            *[
                F.col(column).try_cast("long" if column == "Volume" else "double").alias(column)
                for column in _OHLCV_COLUMNS
            ],
        )

    def _write_to_table(self, data_frame: DataFrame) -> None:
        """Cluster source rows by day before the base class merges them."""
        source = (
            data_frame.withColumn("_day", F.to_date("timestamp"))
            .repartition("_day")
            .sortWithinPartitions("_day", "timestamp", "symbol")
            .drop("_day")
        )
        self._merge_frame(source, self._table_identifier, _MERGE_KEYS)

    def _row_digest(self) -> F.Column:
        """Hash every persisted market-data value for partition completion receipts."""
        return F.sha2(F.to_json(F.struct(*(F.col(column) for column in _CONTENT_HASH_COLUMNS))), 256)

    @property
    def _partition_timestamps(self) -> tuple[datetime, ...]:
        """Return the dataset's normalized requested partition timestamps."""
        return self.config.partition_timestamps

    @property
    def _partition_type(self) -> PartitionType:
        """Return the declared dataset output partition cadence."""
        return self.config.partition_type

    @property
    def _minimum_rows_temporal_filter(self) -> XnysRegularSessionTemporalFilterConfig:
        """Require market-data rows only for partitions overlapping XNYS sessions."""
        return XnysRegularSessionTemporalFilterConfig()

    @property
    def _minimum_rows_reason(self) -> str:
        """Explain the market-data partition completeness requirement."""
        return "Regular XNYS sessions must contain market data."

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

    @property
    def validation_configs(self) -> tuple[ValidationConfig, ...]:
        """Declare the complete dataset output validation policy."""
        return (
            ExactColumnsValidationConfig(
                reason="Market data must use the canonical OHLCV schema.",
                columns=_CONTENT_HASH_COLUMNS,
            ),
            NonNullValuesValidationConfig(
                reason="Market-data fields must be present and castable.",
                columns=_CONTENT_HASH_COLUMNS,
            ),
            FiniteNumericValuesValidationConfig(
                reason="Market prices and volume must be finite.",
                columns=_OHLCV_COLUMNS,
            ),
            UniqueKeyValidationConfig(
                reason="A market-data row is uniquely identified by symbol and timestamp.",
                columns=_MERGE_KEYS,
            ),
        )



    def materialize(self, spark: SparkSession, transform: DatasetTransform) -> DataFrame:
        """Construct the dataset DataFrame for the requested partitions."""
        return transform()


if __name__ == "__main__":
    DatasetMaterializer.main()
