"""Private runtime types shared by data transformations and materializers."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum

from pydantic import Field

from ml_platform.common.types import BaseConfig
from pyspark.sql import DataFrame


class FeatureKey(StrEnum):
    """Supported logical entity keys emitted by data transforms."""

    SYMBOL = "symbol"


class DataType(StrEnum):
    """Supported Iceberg value types emitted by feature and label transforms."""

    DOUBLE = "double"
    FLOAT = "float"
    INT = "int"
    BIGINT = "bigint"


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
        """Return the fixed duration represented by one logical partition."""
        match self:
            case PartitionType.HOURLY:
                return timedelta(hours=1)
            case PartitionType.DAILY:
                return timedelta(days=1)

    @classmethod
    def from_partition_unit(cls, partition_unit: str) -> "PartitionType":
        """Resolve a Spark truncation unit to its explicit partition cadence."""
        match partition_unit:
            case "hour":
                return cls.HOURLY
            case "day":
                return cls.DAILY
            case _ as unsupported_partition_unit:
                raise ValueError(f"Unsupported partition unit: {unsupported_partition_unit}.")

    @classmethod
    def from_iceberg_partition_spec(cls, partition_spec: str) -> "PartitionType":
        """Resolve an Iceberg timestamp transform to its explicit cadence."""
        match partition_spec:
            case "hours(timestamp)":
                return cls.HOURLY
            case "days(timestamp)":
                return cls.DAILY
            case _ as unsupported_partition_spec:
                raise ValueError(f"Unsupported Iceberg partition transform: {unsupported_partition_spec}.")

    def normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize one timestamp to this cadence's UTC partition boundary."""
        normalized = timestamp.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if self is PartitionType.DAILY:
            return normalized.replace(hour=0)
        return normalized


class DatasetDependency(BaseConfig):
    """Immutable declared dataset dependency for a derived materialization."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Stable lowercase name of the upstream dataset.")
    version: int = Field(ge=1, description="Immutable version of the upstream dataset.")
    num_lookback_partitions: int = Field(
        ge=0, description="Number of preceding dataset partitions required for each output partition."
    )

    @property
    def id(self) -> str:
        """Return the canonical resource ID."""
        return f"{self.name}_v{self.version}"


class FeatureDatasetDependency(DatasetDependency):
    """A declared dataset mapped to one feature transform constructor argument."""

    argument_name: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Feature transform constructor argument receiving this dataset.",
    )


@dataclass(frozen=True)
class DatasetReference:
    """Immutable dataset identity and dependency range injected into a transform."""

    name: str
    version: int
    num_lookback_partitions: int

    @property
    def id(self) -> str:
        """Return the canonical resource ID."""
        return f"{self.name}_v{self.version}"


@dataclass(frozen=True)
class Dataset:
    """A dataset reference together with its dependency-window DataFrame."""

    metadata: DatasetReference
    frame: DataFrame
