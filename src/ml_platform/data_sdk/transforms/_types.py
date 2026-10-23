"""Private runtime types shared by data transformations and materializers."""

from dataclasses import dataclass

from pyspark.sql import DataFrame

from ml_platform.common.domain import (
    DatasetDependency,
    FeatureDatasetDependency,
    FeatureInputWindow,
    LabelDatasetDependency,
    LabelInputWindow,
)
from ml_platform.common.types import (
    DatasetId,
    DatasetName,
    DatasetType,
    DataType,
    EntityKey,
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
