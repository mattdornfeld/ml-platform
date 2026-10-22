"""Private runtime types shared by data transformations and materializers."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import assert_never

from pydantic import Field
from pyspark.sql import DataFrame

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk._exceptions import DataSdkError


class FeatureKey(StrEnum):
    """Supported logical entity keys emitted by data transforms."""

    SYMBOL = "symbol"


class PartitionTypeResolutionError(DataSdkError):
    """Raised when an unknown partition unit or Iceberg transform is requested."""


class DatasetType(StrEnum):
    """Dataset transform kinds supported by the materialization system."""

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
                raise PartitionTypeResolutionError(
                    f"Unsupported Spark partition unit {unsupported_partition_unit!r}; expected 'hour' or 'day'."
                )

    @classmethod
    def from_iceberg_partition_spec(
        cls, partition_spec: str
    ) -> "PartitionType":
        """Resolve an Iceberg timestamp transform to its explicit cadence."""
        match partition_spec:
            case "hours(timestamp)":
                return cls.HOURLY
            case "days(timestamp)":
                return cls.DAILY
            case _ as unsupported_partition_spec:
                raise PartitionTypeResolutionError(
                    "Unsupported Iceberg timestamp partition transform "
                    f"{unsupported_partition_spec!r}; expected hours(timestamp) or days(timestamp)."
                )

    def normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize one timestamp to this cadence's UTC partition boundary."""
        normalized = timestamp.astimezone(UTC).replace(
            minute=0, second=0, microsecond=0
        )
        if self is PartitionType.DAILY:
            return normalized.replace(hour=0)
        return normalized


class DatasetDependency(BaseConfig):
    """Immutable declared dataset identity for a derived materialization."""

    name: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase name of the upstream dataset.",
    )
    version: int = Field(
        ge=1, description="Immutable version of the upstream dataset."
    )
    dataset_type: DatasetType = Field(
        description="Kind of upstream dataset used to resolve its Iceberg namespace."
    )
    argument_name: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Transform constructor argument receiving this dataset.",
    )

    @property
    def id(self) -> str:
        """Return the canonical resource ID."""
        return f"{self.name}_v{self.version}"


class FeatureInputWindow(BaseConfig):
    """Historical source partitions available to a point-in-time feature."""

    lookback_partitions: int = Field(
        ge=0,
        description="Number of preceding source partitions required by the feature.",
    )


class LabelInputWindow(FeatureInputWindow):
    """Historical and realized-outcome partitions required by one label."""

    lookahead_partitions: int = Field(
        ge=0,
        description="Number of following source partitions required by the label.",
    )


class FeatureDatasetDependency(DatasetDependency):
    """A point-in-time dataset dependency for a feature transform."""

    input_window: FeatureInputWindow = Field(
        description="Historical source window available to this feature."
    )


class LabelDatasetDependency(DatasetDependency):
    """A realized-outcome dataset dependency for a label transform."""

    input_window: LabelInputWindow = Field(
        description="Historical and future source windows required by this label."
    )


@dataclass(frozen=True)
class DatasetReference:
    """Immutable dataset identity and dependency range injected into a transform."""

    name: str
    version: int

    @property
    def id(self) -> str:
        """Return the canonical resource ID."""
        return f"{self.name}_v{self.version}"


@dataclass(frozen=True)
class Dataset:
    """A dataset reference together with its dependency-window DataFrame."""

    metadata: DatasetReference
    frame: DataFrame
