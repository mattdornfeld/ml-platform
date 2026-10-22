"""Strict Pydantic configuration ontology for data resources."""

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk._exceptions import DataSdkError, MarketSymbolConfigurationError
from ml_platform.data_sdk.materializers._materialization_validator import DiscriminatedValidationConfig
from ml_platform.data_sdk.transforms._types import (
    DataType,
    DatasetType,
    FeatureDatasetDependency,
    FeatureKey,
    PartitionType,
)


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

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Stable lowercase resource name.")
    version: int = Field(ge=1, description="Positive immutable resource-definition version.")
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
    keys: list[FeatureKey] = Field(
        min_length=1,
        description="Ordered logical entity-key columns that identify each dataset row.",
    )
    num_lookback_partitions: int = Field(
        default=0,
        ge=0,
        description="Number of preceding partitions required when this dataset is a dependency.",
    )

    @model_validator(mode="before")
    @classmethod
    def infer_partition_type(cls, values: object) -> object:
        """Use the configured run cadence when no partition type is supplied."""
        if not isinstance(values, dict) or "partition_type" in values:
            return values
        run_frequency = values.get("run_frequency", RunFrequency.DAILY)
        if isinstance(run_frequency, RunFrequency):
            return values | {"partition_type": PartitionType(run_frequency.value)}
        return values | {"partition_type": run_frequency}


class FeatureDatasetConfig(FeatureDatasetDependency):
    """User-facing dependency mapped to a transform constructor argument."""


class TransformConfig(DataConfig):
    """Common typed transformation definition shared by features and labels."""

    dtype: Literal["double", "float", "int", "bigint"] = Field(
        description="Spark SQL type used to persist the transform value."
    )
    frequency: Literal["hourly", "daily"] = Field(
        description="Cadence of the transform's materialized partitions."
    )
    feature_keys: list[FeatureKey] = Field(
        default_factory=lambda: [FeatureKey.SYMBOL],
        min_length=1,
        description="Ordered entity-key columns that identify each transform row.",
    )
    datasets: list[FeatureDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets mapped to transform constructor arguments.",
    )
    transform: str = Field(
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
        description="Absolute import path of the transform implementation.",
    )
    parameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict,
        description="Typed keyword arguments passed to the transform constructor.",
    )

    @property
    def partition_type(self) -> PartitionType:
        """Return the transform's physical cadence derived from its frequency."""
        return PartitionType(self.frequency)

    @model_validator(mode="after")
    def validate_transform_dependencies(self) -> "TransformConfig":
        """Require unambiguous transform identities and injected datasets."""
        if len({dataset.id for dataset in self.datasets}) != len(self.datasets):
            raise TransformDependencyConfigurationError(
                "Transform datasets must have unique name/version identities; "
                f"received: {', '.join(dataset.id for dataset in self.datasets)}."
            )
        if len({dataset.argument_name for dataset in self.datasets}) != len(self.datasets):
            raise TransformDependencyConfigurationError(
                "Transform datasets must have unique constructor arguments; "
                f"received: {', '.join(dataset.argument_name for dataset in self.datasets)}."
            )
        if len(set(self.feature_keys)) != len(self.feature_keys):
            raise TransformDependencyConfigurationError(
                "Transform feature keys must be unique; "
                f"received: {', '.join(feature_key.value for feature_key in self.feature_keys)}."
            )
        return self


class LabelConfig(TransformConfig):
    """Configuration for a materializable label resource."""

    dtype: Literal["double", "float", "int", "bigint"] = Field(
        default="int",
        description="Spark SQL type used to store the label value.",
    )
    frequency: Literal["hourly", "daily"] = Field(
        default="daily",
        description="Cadence of the label's materialized partitions.",
    )


class FeatureConfig(TransformConfig):
    """Configuration for a materializable feature resource."""


class MarketDataConfig(DatasetConfig):
    """Yahoo Finance hourly OHLCV dataset definition."""

    dataset_type: Literal[DatasetType.MARKET_DATA] = Field(
        description="Dataset implementation selected by this definition.")
    symbols: list[str] = Field(min_length=1, description="Yahoo Finance ticker symbols to download.")
    time_range: TimeRange = Field(description="Inclusive and exclusive calendar bounds for source data retrieval.")

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, symbols: list[str]) -> list[str]:
        """Normalize and validate the configured Yahoo Finance symbols."""
        normalized_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not normalized_symbols:
            raise MarketSymbolConfigurationError(
                "Market-data configuration requires at least one non-empty ticker symbol."
            )
        return normalized_symbols


class TripleBarrierConfig(LabelConfig):
    """Triple-barrier label strategy and its volatility-scaled parameters."""

    horizon_bars: int = Field(ge=1, description="Maximum number of bars before the vertical barrier expires.")
    volatility_lookback_bars: int = Field(ge=2, description="Trailing bars used to estimate volatility.")
    profit_taking_multiplier: float = Field(gt=0.0,
                                            description="Volatility multiple that defines the profit-taking barrier.")
    stop_loss_multiplier: float = Field(gt=0.0, description="Volatility multiple that defines the stop-loss barrier.")
