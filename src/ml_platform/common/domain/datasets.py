"""Dataset definition models shared by API clients and Spark workers."""

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

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


class MarketDataDatasetConfig(DomainConfig):
    """Typed market-data settings persisted in a dataset definition."""

    description: Description = Field(
        default=Description("Market data."),
        min_length=1,
        description="Human-readable explanation of the dataset.",
    )
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Validation policies applied when materializing the dataset.",
    )
    keys: list[EntityKey] = Field(
        min_length=1,
        description="Entity columns that identify each market-data observation.",
    )
    partition_type: PartitionType = Field(
        description="Iceberg timestamp partition cadence for the dataset.",
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

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, description: str) -> str:
        """Require a meaningful market-data dataset description."""
        if not description.strip():
            raise MarketSymbolConfigurationError(
                "Description must contain non-whitespace text."
            )
        return description

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        """Keep dataset entity keys and validation overrides unambiguous."""
        if len(set(self.keys)) != len(self.keys):
            raise MarketSymbolConfigurationError(
                "Dataset entity keys must be unique."
            )
        validation_types = {validation.type for validation in self.validations}
        if len(validation_types) != len(self.validations):
            raise MarketSymbolConfigurationError(
                "Dataset validation types must be unique."
            )
        return self


class DatasetConfig(DomainConfig):
    """Full user-facing definition for one materializable dataset version."""

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
    partition_type: PartitionType = Field(
        description="Physical timestamp partition cadence for the dataset.",
    )
    keys: list[EntityKey] = Field(
        min_length=1,
        description="Ordered entity-key columns that identify each dataset row.",
    )
    num_lookback_partitions: LookbackPartition = Field(
        default=LookbackPartition(0),
        ge=0,
        description="Number of preceding partitions required when this dataset is a dependency.",
    )
    description: Description = Field(
        default=Description("No description provided."),
        min_length=1,
        description="Human-readable resource description.",
    )
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Data quality validations applied during materialization.",
    )

    @property
    def id(self) -> DatasetId:
        """Return the canonical immutable dataset identifier."""
        return DatasetId(f"{self.name}_v{self.version}")

    @field_validator("keys", mode="before")
    @classmethod
    def parse_keys(cls, keys: list[EntityKey | str]) -> list[EntityKey]:
        """Parse YAML key literals into the closed EntityKey enum."""
        return [EntityKey(key) for key in keys]

    @model_validator(mode="before")
    @classmethod
    def infer_partition_type(cls, values: object) -> object:
        """Use the configured run cadence when partition type is omitted."""
        if not isinstance(values, dict) or "partition_type" in values:
            return values
        run_frequency = values.get("run_frequency", RunFrequency.DAILY)
        if isinstance(run_frequency, RunFrequency):
            return values | {
                "partition_type": PartitionType(run_frequency.value)
            }
        return values | {"partition_type": run_frequency}


class MarketDataConfig(DatasetConfig):
    """Full YAML definition for a Yahoo Finance OHLCV dataset."""

    dataset_type: Literal[DatasetType.MARKET_DATA] = Field(
        description="Dataset implementation selected by this definition."
    )
    symbols: list[Symbol] = Field(
        min_length=1, description="Yahoo Finance ticker symbols to download."
    )

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, symbols: list[Symbol]) -> list[Symbol]:
        """Normalize and validate the configured Yahoo Finance symbols."""
        normalized_symbols = [
            Symbol(symbol.strip().upper())
            for symbol in symbols
            if symbol.strip()
        ]
        if not normalized_symbols:
            raise MarketSymbolConfigurationError(
                "Market-data configuration requires at least one non-empty ticker symbol."
            )
        return normalized_symbols
