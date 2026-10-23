"""Strict Pydantic configuration ontology for data resources."""

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk._exceptions import (
    DataSdkError,
    MarketSymbolConfigurationError,
)
from ml_platform.data_sdk.materializers._materialization_validator import (
    DiscriminatedValidationConfig,
)
from ml_platform.data_sdk.transforms._types import (
    DatasetDependency,
    DatasetType,
    EntityKey,
    FeatureDatasetDependency,
    LabelDatasetDependency,
    PartitionType,
)
from ml_platform.data_sdk.transforms._types import DataType as DataType

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
    "TimeRange",
    "TransformConfig",
    "TransformName",
    "TripleBarrierConfig",
]

type ParameterKey = str
type ParameterValue = int | float | str | bool
type TransformName = str


class RunFrequency(StrEnum):
    HOURLY = "hourly"
    DAILY = "daily"


class InvalidTimeRangeError(DataSdkError):
    """Raised when a dataset time range does not advance from start to end."""


class TransformDependencyConfigurationError(DataSdkError):
    """Raised when transform input datasets or keys are ambiguous."""


class TimeRange(BaseConfig):
    """Inclusive start and exclusive end dates for a dataset definition."""

    start_date: date = Field(description="First included UTC calendar date.")
    end_date: date = Field(description="First excluded UTC calendar date.")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_yaml_date(cls, value: date | str) -> date:
        """Parse ISO-8601 YAML date values before strict model validation."""
        return date.fromisoformat(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_date_range(self) -> "TimeRange":
        """Require an end date after the required inclusive start date."""
        if self.end_date <= self.start_date:
            raise InvalidTimeRangeError(
                f"Dataset time range end date {self.end_date.isoformat()} must be after "
                f"start date {self.start_date.isoformat()}."
            )
        return self


class DataConfig(BaseConfig):
    """Common immutable metadata for every versioned resource definition."""

    name: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase resource name.",
    )
    version: int = Field(
        ge=1, description="Positive immutable resource-definition version."
    )
    description: str = Field(
        default="No description provided.",
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


class DatasetConfig(DataConfig):
    """Configuration for a materializable dataset definition."""

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


class TransformConfig(DataConfig):
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


class LabelConfig(TransformConfig):
    """Configuration for a materializable label resource."""

    datasets: list[LabelDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into label transforms by dataset name.",
    )


class FeatureConfig(TransformConfig):
    """Configuration for a materializable feature resource."""

    datasets: list[FeatureDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into feature transforms by dataset name.",
    )


class MarketDataConfig(DatasetConfig):
    """Yahoo Finance hourly OHLCV dataset definition."""

    dataset_type: Literal[DatasetType.MARKET_DATA] = Field(
        description="Dataset implementation selected by this definition."
    )
    symbols: list[str] = Field(
        min_length=1, description="Yahoo Finance ticker symbols to download."
    )
    time_range: TimeRange = Field(
        description="Inclusive and exclusive calendar bounds for source data retrieval."
    )

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, symbols: list[str]) -> list[str]:
        """Normalize and validate the configured Yahoo Finance symbols."""
        normalized_symbols = [
            symbol.strip().upper() for symbol in symbols if symbol.strip()
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
