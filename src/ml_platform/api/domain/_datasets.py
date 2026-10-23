"""Dataset definition domain models."""

from dataclasses import dataclass
from datetime import datetime

from ml_platform.api.domain._partitions import PartitionCounts
from ml_platform.common.api.dtos import DatasetStatus
from ml_platform.common.domain import DatasetDefinitionConfig
from ml_platform.common.types import (
    CreatedAt,
    DatasetType,
    ResourceName,
    ResourceVersion,
)


@dataclass(frozen=True)
class DatasetDefinition:
    """A persisted dataset definition independent of HTTP representation."""

    name: ResourceName
    version: ResourceVersion
    dataset_type: DatasetType
    config: DatasetDefinitionConfig
    created_at: CreatedAt
    lifecycle: DatasetStatus
    deleted_at: datetime | None
    deprecated_at: datetime | None


@dataclass(frozen=True)
class DatasetSummary:
    """A dataset definition with its logical partition status totals."""

    definition: DatasetDefinition
    partitions: PartitionCounts


@dataclass(frozen=True)
class DatasetDependency:
    """One transform's declared dependency on a dataset version."""

    name: ResourceName
    version: ResourceVersion
    dataset_type: str
    num_lookback_partitions: int
    num_lookahead_partitions: int
