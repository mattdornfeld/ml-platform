"""Job inspection and materialization response DTOs."""

from enum import StrEnum

from pydantic import Field

from ml_platform.common.api.dtos._base import _ApiModel
from ml_platform.common.api.dtos.partitions import (
    PartitionStatus,
    PartitionUpdateRequestResponse,
)
from ml_platform.common.api.dtos.resources import ResourceType
from ml_platform.common.types import (
    CreatedAt,
    JobId,
    PartitionTimestamp,
    ResourceName,
    ResourceVersion,
    UpdatedAt,
)


class JobType(StrEnum):
    """Kinds of materialization work the control plane can dispatch."""

    DATASET_MATERIALIZATION = "dataset_materialization"
    FEATURE_MATERIALIZATION = "feature_materialization"
    LABEL_MATERIALIZATION = "label_materialization"


class JobStatus(StrEnum):
    """Aggregate lifecycle states for a materialization job."""

    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobTarget(_ApiModel):
    """A versioned resource targeted by a materialization job."""

    resource_type: ResourceType = Field(
        description="Kind of resource targeted by the job.",
    )
    name: ResourceName = Field(description="Stable target resource name.")
    version: ResourceVersion = Field(
        description="Immutable target resource version number.",
    )


class JobPartition(_ApiModel):
    """One resource partition targeted by a materialization job."""

    resource_type: ResourceType = Field(
        description="Kind of resource that owns this partition.",
    )
    name: ResourceName = Field(description="Stable owning resource name.")
    version: ResourceVersion = Field(
        description="Immutable owning resource version number.",
    )
    timestamp: PartitionTimestamp = Field(
        description="UTC start timestamp of the logical partition.",
    )
    status: PartitionStatus = Field(
        description="Current lifecycle state of the requested partition.",
    )


class KubernetesResourceLink(_ApiModel):
    """A read-only Headlamp link for a Kubernetes resource created by a job."""

    resource_type: str = Field(
        description="Kubernetes resource kind, such as SparkApplication or Pod.",
    )
    resource_id: str = Field(description="Kubernetes resource name.")
    url: str = Field(description="Read-only Headlamp URL for the resource.")


class JobResponse(_ApiModel):
    """A materialization job, its targets, partitions, and aggregate lifecycle state."""

    id: JobId = Field(description="Unique materialization job ID.")
    type: JobType = Field(
        description="Materialization work performed by the job."
    )
    status: JobStatus = Field(
        description="Aggregate lifecycle state of the job."
    )
    created_at: CreatedAt = Field(
        description="UTC time when the job was submitted."
    )
    updated_at: UpdatedAt = Field(
        description="UTC time when the job record was last updated.",
    )
    targets: list[JobTarget] = Field(
        default_factory=list,
        description="Versioned resources targeted by the job.",
    )
    partitions: list[JobPartition] = Field(
        default_factory=list,
        description="Logical resource partitions targeted by the job.",
    )
    spark_application_name: str | None = Field(
        default=None,
        description="Kubernetes SparkApplication name, when one was submitted.",
    )
    headlamp_url: str | None = Field(
        default=None,
        description="Read-only Headlamp URL for the SparkApplication, when available.",
    )
    kubernetes_resources: list[KubernetesResourceLink] = Field(
        default_factory=list,
        description="Read-only Headlamp links for Kubernetes resources created by the job.",
    )


class MaterializationJobResponse(_ApiModel):
    """A new job, when submitted, together with every requested partition."""

    submitted_job: JobResponse | None = Field(
        description=(
            "Job created by this request, or null when every requested partition "
            "already exists or is pending."
        )
    )
    partitions: list[PartitionUpdateRequestResponse] = Field(
        description="Partition update requests represented by this submission.",
    )
