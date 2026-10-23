"""Dataset definition models shared by API clients and Spark workers."""

from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from ml_platform.common.domain._base import DomainConfig
from ml_platform.common.domain._exceptions import MarketSymbolConfigurationError
from ml_platform.common.types import (
    DatasetId,
    DatasetName,
    DatasetType,
    Description,
    DiscriminatedValidationConfig,
    EntityKey,
    LookbackPartition,
    PartitionType,
    ResourceVersion,
    RunFrequency,
    Symbol,
)


class DatasetConfig(DomainConfig):
    """Common immutable configuration for every materializable dataset."""

    description: Description = Field(
        default=Description("No description provided."),
        min_length=1,
        description="Human-readable dataset description.",
    )
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Data quality validations applied during materialization.",
    )
    partition_type: PartitionType = Field(
        description="Physical timestamp partition cadence for the dataset.",
    )
    dataset_type: DatasetType = Field(
        description="Discriminator selecting the concrete dataset configuration.",
    )

    @classmethod
    def deserialize(cls, config: dict[str, JsonValue]) -> "DatasetConfigModel":
        """Deserialize persisted configuration with its dataset-type discriminator."""
        return _DATASET_CONFIG_ADAPTER.validate_python(config)

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, description: str) -> str:
        """Require a meaningful dataset description."""
        if not description.strip():
            raise MarketSymbolConfigurationError(
                "Description must contain non-whitespace text."
            )
        return description

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        """Keep dataset validation overrides unambiguous."""
        validation_types = {validation.type for validation in self.validations}
        if len(validation_types) != len(self.validations):
            raise MarketSymbolConfigurationError(
                "Dataset validation types must be unique."
            )
        return self


class MarketDataDatasetConfig(DatasetConfig):
    """Typed Yahoo Finance OHLCV settings for a market-data dataset."""

    dataset_type: Literal[DatasetType.MARKET_DATA] = Field(
        default=DatasetType.MARKET_DATA,
        description="Discriminator selecting the market-data dataset configuration.",
    )
    keys: list[EntityKey] = Field(
        min_length=1,
        description="Entity columns that identify each market-data observation.",
    )
    symbols: list[Symbol] = Field(
        min_length=1,
        description="Yahoo Finance ticker symbols to retrieve.",
    )

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, symbols: list[Symbol]) -> list[Symbol]:
        """Normalize Yahoo Finance symbols and reject blank or duplicates."""
        normalized_symbols = [
            Symbol(symbol.strip().upper()) for symbol in symbols
        ]
        if any(not symbol for symbol in normalized_symbols):
            raise MarketSymbolConfigurationError(
                "Market-data symbols must not be blank."
            )
        if len(set(normalized_symbols)) != len(normalized_symbols):
            raise MarketSymbolConfigurationError(
                "Market-data symbols must be unique."
            )
        return normalized_symbols

    @model_validator(mode="after")
    def validate_market_data_configuration(self) -> Self:
        """Keep market-data entity keys unambiguous."""
        if len(set(self.keys)) != len(self.keys):
            raise MarketSymbolConfigurationError(
                "Dataset entity keys must be unique."
            )
        return self


class MarketDataConfig(MarketDataDatasetConfig):
    """Complete checked-in definition for one Yahoo Finance OHLCV dataset."""

    name: DatasetName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase dataset name.",
    )
    version: ResourceVersion = Field(
        ge=1, description="Positive immutable dataset-definition version."
    )
    run_frequency: RunFrequency = Field(
        default=RunFrequency.DAILY,
        description="Cadence at which the dataset is scheduled to run.",
    )
    num_lookback_partitions: LookbackPartition = Field(
        default=LookbackPartition(0),
        ge=0,
        description="Preceding partitions required when this dataset is an input.",
    )

    @property
    def id(self) -> DatasetId:
        """Return the canonical immutable dataset identifier."""
        return DatasetId(f"{self.name}_v{self.version}")


type DatasetConfigModel = Annotated[
    MarketDataDatasetConfig,
    Field(discriminator="dataset_type"),
]

_DATASET_CONFIG_ADAPTER = TypeAdapter(DatasetConfigModel)
