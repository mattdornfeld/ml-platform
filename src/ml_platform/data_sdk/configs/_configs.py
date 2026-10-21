"""Strict Pydantic configuration ontology for data resources."""

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ml_platform.common.types import BaseConfig


class DataValidations(BaseConfig):
    """Shared validation policy marker for versioned data resources."""


class FeatureKey(StrEnum):
    """Supported logical keys for feature output rows."""

    SYMBOL = "symbol"


class FeatureValidations(DataValidations):
    """Configurable validation requirements for returned feature DataFrames."""

    min_rows_per_partition: int = Field(
        default=1, ge=1, description="Minimum number of rows required for every materialized partition."
    )
    require_finite_feature_value: bool = Field(
        default=True, description="Whether feature values must exclude NaN and positive or negative infinity."
    )
    require_feature_keys_from_input: bool = Field(
        default=True, description="Whether output feature keys must come from the input datasets."
    )


class DatasetType(StrEnum):
    """Dataset types supported by the materialization system."""

    MARKET_DATA = "market_data"


class TimeRange(BaseConfig):
    """Inclusive start and exclusive end dates for a dataset definition."""

    start_date: date = Field(description="First included UTC calendar date.")
    end_date: date = Field(description="First excluded UTC calendar date.")

    @model_validator(mode="after")
    def validate_date_range(self) -> "TimeRange":
        """Require an end date after the required inclusive start date."""
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date.")
        return self


class DataConfig(BaseConfig):
    """Common immutable metadata for a versioned dataset, feature, or label."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Stable lowercase resource name.")
    version: int = Field(ge=1, description="Positive immutable resource-definition version.")
    description: str = Field(default="No description provided.", min_length=1, description="Human-readable resource description.")
    validations: DataValidations = Field(default_factory=DataValidations, description="Data quality validations applied during materialization.")

    @property
    def id(self) -> str:
        """Return the canonical ID derived from name and version."""
        return f"{self.name}_v{self.version}"


class DatasetConfig(DataConfig):
    """Configuration for a dataset definition or a feature dependency."""

    num_lookback_partitions: int = Field(
        default=0, ge=0, description="Number of preceding partitions required when this dataset is a dependency."
    )


class LabelConfig(DataConfig):
    """Base configuration for a materializable label resource."""


class FeatureConfig(DataConfig):
    """Feature metadata and its mapped Spark transformation entrypoint."""

    dtype: Literal["double", "float", "int", "bigint"] = Field(description="Spark SQL type used to store the feature value.")
    frequency: Literal["hourly", "daily"] = Field(description="Cadence of the feature's materialized partitions.")
    feature_key: FeatureKey = Field(default=FeatureKey.SYMBOL, description="Column that identifies the entity for each feature row.")
    datasets: list[DatasetConfig] = Field(min_length=1, description="Versioned datasets read by this feature transform.")
    transform: str = Field(
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
        description="Absolute import path of the FeatureTransform implementation.",
    )
    parameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict, description="Typed keyword arguments passed to the feature transform constructor."
    )
    validations: FeatureValidations = Field(
        default_factory=FeatureValidations, description="Feature-specific data quality validation policy."
    )

    @field_validator("datasets")
    @classmethod
    def require_unique_datasets(cls, datasets: list[DatasetConfig]) -> list[DatasetConfig]:
        """Require at most one declaration for each upstream dataset."""
        if len({dataset.id for dataset in datasets}) != len(datasets):
            raise ValueError("Feature datasets must have unique name and version identities.")
        return datasets


class MarketDataConfig(DatasetConfig):
    """Yahoo Finance hourly OHLCV dataset definition."""

    dataset_type: Literal[DatasetType.MARKET_DATA] = Field(description="Dataset implementation selected by this definition.")
    symbols: list[str] = Field(min_length=1, description="Yahoo Finance ticker symbols to download.")
    time_range: TimeRange = Field(description="Inclusive and exclusive calendar bounds for source data retrieval.")

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, symbols: list[str]) -> list[str]:
        """Normalize and validate the configured Yahoo Finance symbols."""
        normalized_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not normalized_symbols:
            raise ValueError("Provide at least one non-empty market symbol.")
        return normalized_symbols


class TripleBarrierConfig(LabelConfig):
    """Triple-barrier label strategy and its volatility-scaled parameters."""

    horizon_bars: int = Field(ge=1, description="Maximum number of bars before the vertical barrier expires.")
    volatility_lookback_bars: int = Field(ge=2, description="Trailing bars used to estimate volatility.")
    profit_taking_multiplier: float = Field(gt=0.0, description="Volatility multiple that defines the profit-taking barrier.")
    stop_loss_multiplier: float = Field(gt=0.0, description="Volatility multiple that defines the stop-loss barrier.")
