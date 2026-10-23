"""Private runtime types shared by data transformations and materializers."""

from dataclasses import dataclass

from pydantic import Field
from pyspark.sql import DataFrame

from ml_platform.common.types import (
    BaseConfig,
    DatasetId,
    DatasetName,
    DatasetType,
    DataType,
    EntityKey,
    LookAheadPartitions,
    LookbackPartition,
    PartitionType,
    ResourceVersion,
)
from ml_platform.data_sdk._exceptions import DataSdkError

__all__ = [
    "DataType",
    "Dataset",
    "DatasetDependency",
    "DatasetReference",
    "DatasetType",
    "EntityKey",
    "FeatureDatasetDependency",
    "FeatureInputWindow",
    "LabelDatasetDependency",
    "LabelInputWindow",
    "PartitionType",
    "PartitionTypeResolutionError",
]


class PartitionTypeResolutionError(DataSdkError):
    """Raised when an unknown partition transform is requested."""


class DatasetDependency(BaseConfig):
    """Immutable declared dataset identity for a derived materialization."""

    name: DatasetName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase name of the upstream dataset.",
    )
    version: ResourceVersion = Field(
        ge=1, description="Immutable version of the upstream dataset."
    )
    dataset_type: DatasetType = Field(
        description="Kind of upstream dataset used to resolve its Iceberg namespace."
    )

    @property
    def id(self) -> DatasetId:
        """Return the canonical dataset ID."""
        return DatasetId(f"{self.name}_v{self.version}")


class FeatureInputWindow(BaseConfig):
    """Historical source partitions available to a point-in-time feature."""

    lookback_partitions: LookbackPartition = Field(
        ge=0,
        description="Number of preceding source partitions required by the feature.",
    )


class LabelInputWindow(FeatureInputWindow):
    """Historical and realized-outcome partitions required by one label."""

    lookahead_partitions: LookAheadPartitions = Field(
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

    name: DatasetName
    version: ResourceVersion

    @property
    def id(self) -> DatasetId:
        """Return the canonical dataset ID."""
        return DatasetId(f"{self.name}_v{self.version}")


@dataclass(frozen=True)
class Dataset:
    """A dataset reference together with its dependency-window DataFrame."""

    metadata: DatasetReference
    frame: DataFrame
