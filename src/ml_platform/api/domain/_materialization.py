"""Domain models for submitted materialization requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from ml_platform.api.domain import MaterializationJob, PartitionStatus
    from ml_platform.common.types import (
        CreatedAt,
        PartitionId,
        PartitionTimestamp,
        UpdatedAt,
    )


@dataclass(frozen=True)
class MaterializationPartition:
    """One persisted partition update request returned to an API caller."""

    id: PartitionId
    job_id: UUID
    timestamp: PartitionTimestamp
    status: PartitionStatus
    created_at: CreatedAt
    updated_at: UpdatedAt


@dataclass(frozen=True)
class MaterializationRequest:
    """The existing and newly submitted partitions for one request."""

    submitted_job: MaterializationJob | None
    partitions: tuple[MaterializationPartition, ...]
