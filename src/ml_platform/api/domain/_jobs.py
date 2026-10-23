"""API-private immutable domain models for materialization jobs."""

from dataclasses import dataclass
from enum import StrEnum

from ml_platform.api.domain._lifecycle import PartitionStatus
from ml_platform.common.types import (
    CreatedAt,
    JobId,
    PartitionTimestamp,
    ResourceName,
    ResourceVersion,
    UpdatedAt,
)


class JobType(StrEnum):
    """Kinds of work dispatched by the control plane."""

    DATASET_MATERIALIZATION = "dataset_materialization"
    FEATURE_MATERIALIZATION = "feature_materialization"
    LABEL_MATERIALIZATION = "label_materialization"


class JobStatus(StrEnum):
    """Durable lifecycle states for a materialization job."""

    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(StrEnum):
    """Resource kinds materialized by the control plane."""

    DATASET = "dataset"
    FEATURE = "feature"
    LABEL = "label"


@dataclass(frozen=True)
class JobTarget:
    """One immutable resource version targeted by a job."""

    resource_type: ResourceType
    name: ResourceName
    version: ResourceVersion


@dataclass(frozen=True)
class JobPartition:
    """One logical resource partition targeted by a job."""

    resource_type: ResourceType
    name: ResourceName
    version: ResourceVersion
    timestamp: PartitionTimestamp
    status: PartitionStatus


@dataclass(frozen=True)
class MaterializationJob:
    """One persisted materialization job without HTTP presentation details."""

    id: JobId
    type: JobType
    status: JobStatus
    created_at: CreatedAt
    updated_at: UpdatedAt
