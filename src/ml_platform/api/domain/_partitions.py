"""Domain models describing logical partitions and persisted receipts."""

from dataclasses import dataclass
from datetime import datetime

from ml_platform.api.domain._lifecycle import PartitionStatus
from ml_platform.common.types import (
    CreatedAt,
    PartitionId,
    PartitionTimestamp,
    SnapshotId,
    UpdatedAt,
)


@dataclass(frozen=True)
class PartitionCounts:
    """Counts of the latest logical partitions for one resource version."""

    total: int
    created: int
    pending: int
    failed: int


@dataclass(frozen=True)
class ResourcePartition:
    """Latest requested revision for one logical resource partition."""

    id: PartitionId
    timestamp: PartitionTimestamp
    status: PartitionStatus
    created_at: CreatedAt
    updated_at: UpdatedAt
    content_hash: str | None
    snapshot_id: SnapshotId | None


@dataclass(frozen=True)
class PartitionReceipt:
    """The immutable persisted receipt for one completed resource partition."""

    partition_id: PartitionId
    content_hash: str
    snapshot_id: SnapshotId
    row_count: int
    iceberg_schema: dict[str, str]
    created_at: datetime
