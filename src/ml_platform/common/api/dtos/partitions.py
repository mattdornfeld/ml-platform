"""Partition materialization and completion API DTOs."""

from datetime import timedelta
from enum import StrEnum
from typing import Annotated

from pydantic import (
    AfterValidator,
    Field,
    field_validator,
    model_validator,
)

from ml_platform.common.api.dtos._base import _ApiModel
from ml_platform.common.api.dtos.resources import ResourceType
from ml_platform.common.types import (
    CallbackToken,
    ContentHash,
    CreatedAt,
    IcebergSchema,
    JobId,
    PartitionId,
    PartitionTimestamp,
    RowCount,
    SnapshotId,
    UpdatedAt,
)


class PartitionStatus(StrEnum):
    """Lifecycle states for an individual partition update request."""

    PENDING = "pending"
    CREATED = "created"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PartitionMaterializationRequest(_ApiModel):
    """Logical UTC partitions requested for one materialization job."""

    timestamps: list[PartitionTimestamp] = Field(
        min_length=1,
        description="UTC timestamps aligned exactly to the start of an hour.",
        examples=[["2026-10-20T18:00:00Z"]],
    )

    @field_validator("timestamps")
    @classmethod
    def require_hour_aligned_utc_timestamps(
        cls, timestamps: list[PartitionTimestamp]
    ) -> list[PartitionTimestamp]:
        """Require every requested partition timestamp to be an exact UTC hour."""
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("Partition timestamps must be unique.")
        for timestamp in timestamps:
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(
                    "Partition timestamps must include a UTC offset."
                )
            if timestamp.utcoffset() != timedelta(0):
                raise ValueError("Partition timestamps must be in UTC.")
            if any((timestamp.minute, timestamp.second, timestamp.microsecond)):
                raise ValueError(
                    "Partition timestamps must be aligned to an exact hour."
                )
        return timestamps


class PartitionUpdateRequestResponse(_ApiModel):
    """A pending or terminal dataset or feature partition update request."""

    partition_id: PartitionId = Field(
        description="Stable ID for this logical partition update request.",
    )
    job_id: JobId = Field(
        description="Materialization job that owns the request."
    )
    timestamp: PartitionTimestamp = Field(
        description="UTC timestamp aligned exactly to the start of an hour.",
        examples=["2026-10-20T18:00:00Z"],
    )
    status: PartitionStatus = Field(
        description="Current lifecycle state of the partition update request.",
    )
    created_at: CreatedAt = Field(
        description="UTC time when the partition request was created.",
    )
    updated_at: UpdatedAt = Field(
        description="UTC time when the partition request was last updated.",
    )


class PartitionCompletionRequest(_ApiModel):
    """One authenticated receipt in an atomic Iceberg completion batch."""

    partition_id: PartitionId = Field(
        description="ID of the partition update request being completed.",
    )
    resource_type: ResourceType = Field(
        description="Kind of resource that owns the completed partition.",
    )
    row_count: RowCount = Field(
        ge=0,
        description="Number of rows committed for the logical partition.",
    )
    iceberg_schema: IcebergSchema = Field(
        min_length=1,
        description="Column names and Spark SQL types written to the Iceberg table.",
    )
    content_hash: ContentHash = Field(
        min_length=1,
        description="Deterministic hash of the committed partition contents.",
    )
    snapshot_id: SnapshotId = Field(
        ge=0,
        description="Iceberg table snapshot that atomically committed the partition.",
    )
    callback_token: CallbackToken = Field(
        min_length=1,
        description="Opaque token authorizing this individual partition completion.",
    )
    dataset_partition_ids: list[PartitionId] | None = Field(
        default=None,
        description="Pinned upstream dataset partition IDs for derived resources.",
    )

    @field_validator("iceberg_schema")
    @classmethod
    def require_named_schema_fields(
        cls, iceberg_schema: IcebergSchema
    ) -> dict[str, str]:
        """Reject empty column names or type declarations in committed schemas."""
        if any(
            not name.strip() or not dtype.strip()
            for name, dtype in iceberg_schema.items()
        ):
            raise ValueError(
                "Completion schema requires non-empty column names and types."
            )
        return iceberg_schema

    @field_validator("content_hash", "callback_token")
    @classmethod
    def reject_blank_authentication_fields(cls, value: str) -> str:
        """Reject whitespace-only receipt hashes and callback credentials."""
        if not value.strip():
            raise ValueError("Value must contain non-whitespace text.")
        return value

    @model_validator(mode="after")
    def validate_dependencies(self) -> "PartitionCompletionRequest":
        """Derived resources must report their exact pinned dataset revisions."""
        if (
            self.resource_type is ResourceType.DATASET
            and self.dataset_partition_ids is not None
        ):
            raise ValueError(
                "Dataset receipts must not include dataset_partition_ids."
            )
        if self.resource_type is not ResourceType.DATASET:
            if not self.dataset_partition_ids:
                raise ValueError(
                    "Derived receipts require dataset_partition_ids."
                )
            if len(set(self.dataset_partition_ids)) != len(
                self.dataset_partition_ids
            ):
                raise ValueError("dataset_partition_ids must be unique.")
        return self


def _validate_completion_batch(
    receipts: list[PartitionCompletionRequest],
) -> list[PartitionCompletionRequest]:
    """Reject duplicate receipt identities before repository processing."""
    partition_ids = [item.partition_id for item in receipts]
    if len(set(partition_ids)) != len(partition_ids):
        raise ValueError("Completion batches must use unique partition IDs.")
    return receipts


type PartitionCompletionBatch = Annotated[
    list[PartitionCompletionRequest],
    AfterValidator(_validate_completion_batch),
]


class PartitionResultResponse(_ApiModel):
    """The recorded receipt for one successfully materialized partition."""

    partition_id: PartitionId = Field(
        description="ID of the completed partition update request.",
    )
    resource_type: ResourceType = Field(
        description="Kind of resource that owns the completed partition.",
    )
    row_count: RowCount = Field(
        description="Number of rows committed for the logical partition.",
    )
    iceberg_schema: IcebergSchema = Field(
        description="Column names and Spark SQL types in the committed Iceberg table.",
    )
    content_hash: ContentHash = Field(
        description="Deterministic hash of the committed partition contents.",
    )
    snapshot_id: SnapshotId = Field(
        description="Iceberg table snapshot that committed the partition.",
    )
    created_at: CreatedAt = Field(
        description="UTC time at which the completion receipt was recorded.",
    )
