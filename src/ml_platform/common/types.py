"""Shared strict Pydantic and domain types."""

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal, NewType, assert_never
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Distinct NewType wrappers make domain primitives statically non-interchangeable.
CallbackToken = NewType("CallbackToken", str)
ContentHash = NewType("ContentHash", str)
Description = NewType("Description", str)
DatasetId = NewType("DatasetId", str)
DatasetName = NewType("DatasetName", str)
FeatureId = NewType("FeatureId", str)
FeatureName = NewType("FeatureName", str)
LabelId = NewType("LabelId", str)
LabelName = NewType("LabelName", str)
ParameterKey = NewType("ParameterKey", str)
type ParameterValue = int | float | str | bool
PartitionTimestamp = NewType("PartitionTimestamp", datetime)
ResourceName = NewType("ResourceName", str)
ResourceVersion = NewType("ResourceVersion", int)
Symbol = NewType("Symbol", str)
TransformName = NewType("TransformName", str)
IcebergSchema = NewType("IcebergSchema", dict[str, str])
JobId = NewType("JobId", UUID)
PartitionId = NewType("PartitionId", UUID)
CreatedAt = NewType("CreatedAt", datetime)
UpdatedAt = NewType("UpdatedAt", datetime)
LookbackPartition = NewType("LookbackPartition", int)
LookAheadPartitions = NewType("LookAheadPartitions", int)


class EntityKey(StrEnum):
    """Supported logical entity keys emitted by data transforms."""

    SYMBOL = "symbol"


class DatasetType(StrEnum):
    """Dataset transform kinds supported by materialization."""

    MARKET_DATA = "market_data"

    @property
    def iceberg_namespace(self) -> str:
        """Return the namespace assigned to this dataset kind's Iceberg tables."""
        match self:
            case DatasetType.MARKET_DATA:
                return "market_data"
            case _ as unsupported_dataset_type:
                assert_never(unsupported_dataset_type)


class DataType(StrEnum):
    """Supported persisted feature and label value types."""

    DOUBLE = "double"
    FLOAT = "float"
    INT = "int"
    BIGINT = "bigint"


class RunFrequency(StrEnum):
    """Supported configured run frequencies."""

    HOURLY = "hourly"
    DAILY = "daily"


class PartitionType(StrEnum):
    """Supported Iceberg timestamp partition transforms."""

    HOURLY = "hourly"
    DAILY = "daily"

    @property
    def partition_unit(self) -> str:
        """Return the Spark timestamp truncation unit for this cadence."""
        match self:
            case PartitionType.HOURLY:
                return "hour"
            case PartitionType.DAILY:
                return "day"

    @property
    def iceberg_partition_spec(self) -> str:
        """Return the fixed Iceberg partition transform for this cadence."""
        return f"{self.partition_unit}s(timestamp)"

    @property
    def partition_interval(self) -> timedelta:
        """Return the duration represented by one logical partition."""
        match self:
            case PartitionType.HOURLY:
                return timedelta(hours=1)
            case PartitionType.DAILY:
                return timedelta(days=1)

    @classmethod
    def from_partition_unit(cls, partition_unit: str) -> "PartitionType":
        """Resolve a Spark timestamp truncation unit to an explicit cadence."""
        match partition_unit:
            case "hour":
                return cls.HOURLY
            case "day":
                return cls.DAILY
            case _ as unsupported_partition_unit:
                raise ValueError(
                    f"Unsupported Spark partition unit {unsupported_partition_unit!r}."
                )

    @classmethod
    def from_iceberg_partition_spec(
        cls, partition_spec: str
    ) -> "PartitionType":
        """Resolve an Iceberg timestamp transform to an explicit cadence."""
        match partition_spec:
            case "hours(timestamp)":
                return cls.HOURLY
            case "days(timestamp)":
                return cls.DAILY
            case _ as unsupported_partition_spec:
                raise ValueError(
                    "Unsupported Iceberg timestamp partition transform "
                    f"{unsupported_partition_spec!r}."
                )

    def normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize one timestamp to this cadence's UTC partition boundary."""
        normalized = timestamp.astimezone(UTC).replace(
            minute=0, second=0, microsecond=0
        )
        return (
            normalized.replace(hour=0)
            if self is PartitionType.DAILY
            else normalized
        )


class TemporalFilterConfig(BaseModel):
    """Base type for selecting requested partitions during validation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    type: str = Field(description="Discriminator for the temporal filter.")


class AllPartitionsTemporalFilterConfig(TemporalFilterConfig):
    """Apply a partition-level validation to every requested partition."""

    type: Literal["all_partitions"] = Field(
        default="all_partitions",
        description="Apply the validation to every requested partition.",
    )


class XnysRegularSessionTemporalFilterConfig(TemporalFilterConfig):
    """Select partitions that overlap an XNYS regular trading session."""

    type: Literal["xnys_regular_session"] = Field(
        default="xnys_regular_session",
        description="Apply only during XNYS regular trading sessions.",
    )


type TemporalFilter = Annotated[
    AllPartitionsTemporalFilterConfig | XnysRegularSessionTemporalFilterConfig,
    Field(discriminator="type"),
]


class ValidationConfig(BaseModel):
    """Base type for one declared materialization validation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    type: str = Field(description="Discriminator for the validation.")
    reason: str = Field(
        min_length=1,
        description="Why this validation is required for materialized data.",
    )


class PartitionValidationConfig(ValidationConfig):
    """Validation applied to selected requested output partitions."""

    temporal_filter: TemporalFilter = Field(
        default_factory=AllPartitionsTemporalFilterConfig,
        description="Requested output partitions to validate.",
    )


class MinimumRowsValidationConfig(PartitionValidationConfig):
    """Require a minimum number of output rows per selected partition."""

    type: Literal["minimum_rows"] = Field(
        default="minimum_rows", description="Require a minimum row count."
    )
    minimum_rows: int = Field(
        ge=1,
        description="Minimum output rows required in each selected partition.",
    )
    timestamps: tuple[datetime, ...] = Field(
        default=(),
        exclude=True,
        description="UTC output partition boundaries to check.",
    )
    partition_type: PartitionType = Field(
        default=PartitionType.DAILY,
        exclude=True,
        description="Explicit cadence of the checked partitions.",
    )
    timestamp_column: str = Field(
        default="timestamp",
        exclude=True,
        description="Output timestamp column name.",
    )
    partition_column: str = Field(
        default="output_partition",
        exclude=True,
        description="Temporary logical partition column name.",
    )


class NonNullValuesValidationConfig(ValidationConfig):
    """Require declared persisted output columns to be non-null."""

    type: Literal["non_null_values"] = Field(
        default="non_null_values", description="Require non-null output values."
    )
    columns: tuple[str, ...] = Field(
        default=(), description="Output columns that must be non-null."
    )


class FiniteNumericValuesValidationConfig(ValidationConfig):
    """Require declared numeric output columns to be finite."""

    type: Literal["finite_numeric_values"] = Field(
        default="finite_numeric_values",
        description="Require finite numeric output values.",
    )
    columns: tuple[str, ...] = Field(
        default=(), description="Output numeric columns that must be finite."
    )


class KeysFromInputValidationConfig(ValidationConfig):
    """Require entity keys to occur in injected input datasets."""

    type: Literal["keys_from_input"] = Field(
        default="keys_from_input",
        description="Require entity keys from input datasets.",
    )
    keys: tuple[str, ...] = Field(
        default=(), description="Output entity-key columns to validate."
    )
    dataset_table_identifiers: tuple[str, ...] = Field(
        default=(),
        description="Iceberg input tables that define valid entity keys.",
    )


class ExactColumnsValidationConfig(ValidationConfig):
    """Require an exact output-column set."""

    type: Literal["exact_columns"] = Field(
        default="exact_columns",
        description="Require the exact output-column set.",
    )
    columns: tuple[str, ...] = Field(
        default=(), description="Complete required output column set."
    )


class UniqueKeyValidationConfig(ValidationConfig):
    """Require each declared output key tuple to be unique."""

    type: Literal["unique_key"] = Field(
        default="unique_key", description="Require a unique output key tuple."
    )
    columns: tuple[str, ...] = Field(
        default=(),
        description="Columns that together identify a unique output row.",
    )


class RequestedPartitionsValidationConfig(ValidationConfig):
    """Restrict output rows to declared timestamp partition boundaries."""

    type: Literal["requested_partitions"] = Field(
        default="requested_partitions",
        description="Restrict output to requested partitions.",
    )
    timestamps: tuple[datetime, ...] = Field(
        default=(),
        exclude=True,
        description="UTC output partition boundaries allowed in the data frame.",
    )
    partition_type: PartitionType = Field(
        default=PartitionType.DAILY,
        exclude=True,
        description="Explicit cadence of the checked partitions.",
    )
    timestamp_column: str = Field(
        default="timestamp",
        exclude=True,
        description="Output timestamp column name.",
    )
    partition_column: str = Field(
        default="output_partition",
        exclude=True,
        description="Temporary logical partition column name.",
    )


type DiscriminatedValidationConfig = Annotated[
    MinimumRowsValidationConfig
    | NonNullValuesValidationConfig
    | FiniteNumericValuesValidationConfig
    | KeysFromInputValidationConfig
    | ExactColumnsValidationConfig
    | UniqueKeyValidationConfig
    | RequestedPartitionsValidationConfig,
    Field(discriminator="type"),
]


class BaseConfig(BaseModel):
    """Immutable, strict base for ML Platform Pydantic configuration models."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        revalidate_instances="always",
        arbitrary_types_allowed=False,
        allow_inf_nan=False,
    )
