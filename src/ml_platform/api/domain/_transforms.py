"""Feature and label transform definition domain models."""

from dataclasses import dataclass
from datetime import datetime

from ml_platform.api.domain._jobs import MaterializationJob
from ml_platform.api.domain._lifecycle import ResourceLifecycle
from ml_platform.api.domain._partitions import (
    PartitionCounts,
    ResourcePartition,
)
from ml_platform.common.types import (
    CreatedAt,
    DatasetName,
    DatasetType,
    Description,
    ResourceName,
    ResourceVersion,
)


@dataclass(frozen=True)
class TransformDefinition:
    """A persisted feature or label transform definition."""

    name: ResourceName
    version: ResourceVersion
    dtype: str
    frequency: str
    keys: tuple[str, ...]
    validations: tuple[dict[str, object], ...]
    transform: str
    description: Description
    parameters: dict[str, object]
    created_at: CreatedAt
    lifecycle: ResourceLifecycle
    deleted_at: datetime | None
    deprecated_at: datetime | None


@dataclass(frozen=True)
class TransformSummary:
    """A transform definition with its logical partition status totals."""

    definition: TransformDefinition
    partitions: PartitionCounts


@dataclass(frozen=True)
class TransformDatasetDependency:
    """One versioned dataset consumed by a transform."""

    name: DatasetName
    version: ResourceVersion
    dataset_type: DatasetType
    lookback_partitions: int
    lookahead_partitions: int


@dataclass(frozen=True)
class TransformDetail:
    """One transform definition with all displayable persisted state."""

    definition: TransformDefinition
    datasets: tuple[TransformDatasetDependency, ...]
    partitions: tuple[ResourcePartition, ...]
    jobs: tuple[MaterializationJob, ...]
