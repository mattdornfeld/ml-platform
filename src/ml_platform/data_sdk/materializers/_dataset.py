"""API-dispatched Spark materializer for market-data datasets."""

from dataclasses import dataclass

from pydantic import Field, field_validator
from pyspark.sql import SparkSession
from pyspark.sql import functions as spark_functions
from pyspark.sql.column import Column

from ml_platform.data_sdk._exceptions import MarketSymbolConfigurationError
from ml_platform.data_sdk.materializers._base import (
    Materializer,
    MaterializerConfig,
)
from ml_platform.data_sdk.materializers._iceberg_table import IcebergTable
from ml_platform.data_sdk.materializers._materialization_validator import (
    ExactColumnsValidationConfig,
    FiniteNumericValuesValidationConfig,
    MinimumRowsValidationConfig,
    NonNullValuesValidationConfig,
    UniqueKeyValidationConfig,
    ValidationConfig,
    XnysRegularSessionTemporalFilterConfig,
)
from ml_platform.data_sdk.materializers.exceptions import (
    UnsupportedDatasetTypeError,
)
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.transforms._types import (
    DatasetType,
    FeatureKey,
    PartitionType,
)

_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_CONTENT_HASH_COLUMNS = (
    "timestamp",
    "symbol",
    "market_session",
    *_OHLCV_COLUMNS,
)


class DatasetMaterializationConfig(MaterializerConfig):
    """Complete JSON contract consumed by a dataset Spark materializer."""

    dataset_type: DatasetType = Field(
        description="Kind of dataset transform to execute."
    )
    keys: list[FeatureKey] = Field(
        min_length=1,
        description="Ordered logical entity keys emitted by the dataset transform.",
    )
    partition_type: PartitionType = Field(
        description="Iceberg timestamp partition cadence used for this dataset.",
    )
    symbols: list[str] = Field(
        min_length=1,
        description="Yahoo Finance ticker symbols to retrieve for each requested partition.",
    )

    @field_validator("symbols")
    @classmethod
    def _normalize_symbols(cls, symbols: list[str]) -> list[str]:
        normalized = [
            symbol.strip().upper() for symbol in symbols if symbol.strip()
        ]
        if not normalized:
            raise MarketSymbolConfigurationError(
                "Market-data materialization requires at least one non-empty ticker symbol."
            )
        return normalized


@dataclass(frozen=True)
class DatasetMaterializer(Materializer[DatasetMaterializationConfig]):
    """Construct, validate, persist, and report one market-data materialization job."""

    @property
    def _partition_type(self) -> PartitionType:
        """Return the market-data output cadence."""
        return self.config.partition_type

    @property
    def iceberg_table(self) -> IcebergTable:
        """Return the immutable market-data table definition."""
        return IcebergTable(
            catalog=self.config.iceberg_catalog(),
            namespace="market_data",
            name=self.config.name,
            version=self.config.version,
            resource_type="dataset",
            schema={
                "timestamp": "timestamp",
                "symbol": "string",
                "market_session": "string",
                "Open": "double",
                "High": "double",
                "Low": "double",
                "Close": "double",
                "Volume": "bigint",
            },
            partition_type=self.config.partition_type,
            description=self.config.description,
            dependencies=(),
            validations=tuple(
                validation.model_dump(mode="json")
                for validation in self.config.validations
            ),
            additional_properties=(),
        )

    @classmethod
    def config_model(cls) -> type[DatasetMaterializationConfig]:
        """Return the strict runtime contract for market-data jobs."""
        return DatasetMaterializationConfig

    @classmethod
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for market-data jobs."""
        return "ml-platform-dataset-materializer"

    def _row_digest(self) -> Column:
        """Hash every persisted market-data value for partition completion receipts."""
        return spark_functions.sha2(
            spark_functions.to_json(
                spark_functions.struct(
                    *(
                        spark_functions.col(column)
                        for column in _CONTENT_HASH_COLUMNS
                    )
                )
            ),
            256,
        )

    def _transform(self, spark: SparkSession) -> DatasetTransform:
        """Build the user-facing transform selected by the dataset definition."""
        match self.config.dataset_type:
            case DatasetType.MARKET_DATA:
                from ml_platform.data_sdk.dataset_transforms._market_data import (
                    MarketDataTransform,
                )

                return MarketDataTransform(
                    spark=spark,
                    symbols=self.config.symbols,
                    partitions=[
                        partition.timestamp
                        for partition in self.config.partitions
                    ],
                    keys=self.config.keys,
                    partition_type=self.config.partition_type,
                )
            case _ as unsupported_type:
                raise UnsupportedDatasetTypeError(
                    f"Dataset materializer does not support dataset type {unsupported_type!r}."
                )

    @property
    def validation_configs(self) -> tuple[ValidationConfig, ...]:
        """Declare the complete dataset output validation policy."""
        return (
            MinimumRowsValidationConfig(
                reason="Regular XNYS sessions must contain market data.",
                minimum_rows=1,
                timestamps=self.config.partition_timestamps,
                partition_type=self.config.partition_type,
                temporal_filter=XnysRegularSessionTemporalFilterConfig(),
            ),
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
                columns=("symbol", "timestamp"),
            ),
        )


if __name__ == "__main__":
    DatasetMaterializer.main()
