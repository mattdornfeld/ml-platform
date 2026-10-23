"""Strict Pydantic configuration ontology for data resources."""

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from ml_platform.common.types import (
    BaseConfig,
    DatasetName,
    DatasetType,
    DataType,
    Description,
    DiscriminatedValidationConfig,
    EntityKey,
    FeatureName,
    LabelName,
    ParameterKey,
    ParameterValue,
    PartitionType,
    ResourceVersion,
    RunFrequency,
    Symbol,
    TransformName,
)
from ml_platform.data_sdk._exceptions import (
    DataSdkError,
    MarketSymbolConfigurationError,
)
from ml_platform.data_sdk.transforms._types import (
    DatasetDependency,
    FeatureDatasetDependency,
    LabelDatasetDependency,
)

__all__ = [
    "BaseConfig",
    "DataConfig",
    "DataType",
    "DatasetConfig",
    "DatasetType",
    "EntityKey",
    "FeatureConfig",
    "LabelConfig",
    "LabelDatasetDependency",
    "MarketDataConfig",
    "ParameterKey",
    "ParameterValue",
    "PartitionType",
    "RunFrequency",
    "TransformConfig",
    "TransformName",
    "TripleBarrierConfig",
]


class TransformDependencyConfigurationError(DataSdkError):
    """Raised when transform input datasets or keys are ambiguous."""


class DataConfig[NameT: str](BaseConfig):
    """Common immutable metadata for every versioned resource definition."""

    name: NameT = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase resource name.",
    )
    version: ResourceVersion = Field(
        ge=1, description="Positive immutable resource-definition version."
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
    def id(self) -> str:
        """Return the canonical ID derived from name and version."""
        return f"{self.name}_v{self.version}"


class DatasetConfig(DataConfig[DatasetName]):
    """Configuration for a materializable dataset definition."""

    name: DatasetName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase dataset name.",
    )

    run_frequency: RunFrequency = Field(
        default=RunFrequency.DAILY,
        description="Cadence at which the dataset is scheduled to run.",
    )
    partition_type: PartitionType = Field(
        description="Physical timestamp partition cadence; defaults to the configured run frequency."
    )
    keys: list[EntityKey] = Field(
        min_length=1,
        description="Ordered logical entity-key columns that identify each dataset row.",
    )
    num_lookback_partitions: int = Field(
        default=0,
        ge=0,
        description="Number of preceding partitions required when this dataset is a dependency.",
    )

    @field_validator("keys", mode="before")
    @classmethod
    def parse_keys(cls, keys: list[EntityKey | str]) -> list[EntityKey]:
        """Parse YAML key literals into the closed EntityKey enum."""
        return [EntityKey(key) for key in keys]

    @model_validator(mode="before")
    @classmethod
    def infer_partition_type(cls, values: object) -> object:
        """Use the configured run cadence when no partition type is supplied."""
        if not isinstance(values, dict) or "partition_type" in values:
            return values
        run_frequency = values.get("run_frequency", RunFrequency.DAILY)
        if isinstance(run_frequency, RunFrequency):
            return values | {
                "partition_type": PartitionType(run_frequency.value)
            }
        return values | {"partition_type": run_frequency}


class TransformConfig[NameT: str](DataConfig[NameT]):
    """Common typed transformation definition shared by features and labels."""

    dtype: DataType = Field(
        description="Spark SQL type used to persist the transform value."
    )
    frequency: RunFrequency = Field(
        description="Cadence of the transform's materialized partitions."
    )
    keys: list[EntityKey] = Field(
        default_factory=lambda: [EntityKey.SYMBOL],
        min_length=1,
        description="Ordered entity-key columns that identify each transform row.",
    )
    datasets: Sequence[DatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into transforms by dataset name.",
    )
    transform: TransformName = Field(
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
        description="Absolute import path of the transform implementation.",
    )
    parameters: dict[ParameterKey, ParameterValue] = Field(
        default_factory=dict,
        description="Typed keyword arguments passed to the transform constructor.",
    )

    @field_validator("keys", mode="before")
    @classmethod
    def parse_keys(cls, keys: list[EntityKey | str]) -> list[EntityKey]:
        """Parse YAML key literals into the closed EntityKey enum."""
        return [EntityKey(feature_key) for feature_key in keys]

    @property
    def partition_type(self) -> PartitionType:
        """Return the transform's physical cadence derived from its frequency."""
        return PartitionType(self.frequency)

    @model_validator(mode="after")
    def validate_transform_dependencies(self) -> Self:
        """Require unambiguous transform identities and injected datasets."""
        if len({dataset.id for dataset in self.datasets}) != len(self.datasets):
            raise TransformDependencyConfigurationError(
                "Transform datasets must have unique name/version identities; "
                f"received: {', '.join(dataset.id for dataset in self.datasets)}."
            )
        if len({dataset.name for dataset in self.datasets}) != len(
            self.datasets
        ):
            raise TransformDependencyConfigurationError(
                "Transform datasets must have unique names because dataset names "
                "are transform constructor arguments; "
                f"received: {', '.join(dataset.name for dataset in self.datasets)}."
            )
        if len(set(self.keys)) != len(self.keys):
            raise TransformDependencyConfigurationError(
                "Transform entity keys must be unique; "
                f"received: {', '.join(feature_key.value for feature_key in self.keys)}."
            )
        return self


class LabelConfig(TransformConfig[LabelName]):
    """Configuration for a materializable label resource."""

    name: LabelName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase label name.",
    )

    datasets: list[LabelDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into label transforms by dataset name.",
    )


class FeatureConfig(TransformConfig[FeatureName]):
    """Configuration for a materializable feature resource."""

    name: FeatureName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase feature name.",
    )

    datasets: list[FeatureDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into feature transforms by dataset name.",
    )


class MarketDataConfig(DatasetConfig):
    """Yahoo Finance hourly OHLCV dataset definition."""

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


class TripleBarrierConfig(LabelConfig):
    """Triple-barrier label strategy and its volatility-scaled parameters."""

    horizon_bars: int = Field(
        ge=1,
        description="Maximum number of bars before the vertical barrier expires.",
    )
    volatility_lookback_bars: int = Field(
        ge=2, description="Trailing bars used to estimate volatility."
    )
    profit_taking_multiplier: float = Field(
        gt=0.0,
        description="Volatility multiple that defines the profit-taking barrier.",
    )
    stop_loss_multiplier: float = Field(
        gt=0.0,
        description="Volatility multiple that defines the stop-loss barrier.",
    )
