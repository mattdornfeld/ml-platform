"""Typed upstream-dataset dependency declarations."""

from pydantic import Field

from ml_platform.common.domain._base import DomainConfig
from ml_platform.common.types import (
    DatasetId,
    DatasetName,
    DatasetType,
    LookAheadPartitions,
    LookbackPartition,
    ResourceVersion,
)


class DatasetDependency(DomainConfig):
    """Immutable identity of one dataset consumed by a derived resource."""

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
        """Return the canonical immutable dataset identifier."""
        return DatasetId(f"{self.name}_v{self.version}")


class FeatureInputWindow(DomainConfig):
    """Historical source partitions available to one point-in-time feature."""

    lookback_partitions: LookbackPartition = Field(
        ge=0,
        description="Number of preceding source partitions required by the feature.",
    )


class LabelInputWindow(FeatureInputWindow):
    """Historical and realized-outcome source partitions required by one label."""

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
        description="Historical and future source windows required by the label."
    )
