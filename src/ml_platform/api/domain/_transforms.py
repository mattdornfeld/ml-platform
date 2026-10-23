"""Feature and label transform definition domain models."""

from dataclasses import dataclass
from datetime import datetime

from ml_platform.api.domain._lifecycle import ResourceLifecycle
from ml_platform.api.domain._partitions import PartitionCounts
from ml_platform.common.types import (
    CreatedAt,
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
