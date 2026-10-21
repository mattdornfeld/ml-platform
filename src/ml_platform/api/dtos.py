"""HTTP request and response models for the materialization control plane."""

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobType(StrEnum):
    """Kinds of materialization work the control plane can dispatch."""

    DATASET_MATERIALIZATION = "dataset_materialization"
    FEATURE_MATERIALIZATION = "feature_materialization"


class JobStatus(StrEnum):
    """Aggregate lifecycle states for a materialization job."""

    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PartitionStatus(StrEnum):
    """Lifecycle states for an individual partition update request."""

    PENDING = "pending"
    CREATED = "created"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeatureKey(StrEnum):
    """Supported logical keys for feature output rows."""

    SYMBOL = "symbol"


class FeatureValidations(BaseModel):
    """Configurable validation requirements for returned feature DataFrames."""

    model_config = ConfigDict(extra="forbid")

    min_rows_per_partition: int = Field(default=1, ge=1)
    require_finite_feature_value: bool = True
    require_feature_keys_from_input: bool = True


class DatasetType(StrEnum):
    """Dataset types accepted by the control plane."""

    MARKET_DATA = "market_data"


class DatasetCreateRequest(BaseModel):
    """Payload for registering an immutable dataset definition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    dataset_type: DatasetType
    config: dict[str, Any]


class DatasetResponse(DatasetCreateRequest):
    """A persisted dataset definition."""

    created_at: datetime
    job_id: UUID | None = None


class FeatureDatasetDependency(BaseModel):
    """A registered dataset version consumed by a feature."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    num_lookback_partitions: int = Field(ge=0)


class FeatureCreateRequest(BaseModel):
    """Payload for registering an immutable, normalized feature definition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    dtype: Literal["double", "float", "int", "bigint"]
    frequency: Literal["hourly", "daily"]
    feature_key: FeatureKey = FeatureKey.SYMBOL
    datasets: list[FeatureDatasetDependency] = Field(min_length=1)
    transform: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    validations: FeatureValidations = Field(default_factory=FeatureValidations)


class FeatureResponse(FeatureCreateRequest):
    """A persisted feature definition."""

    created_at: datetime


class FeatureMaterializationMode(StrEnum):
    """How a feature materialization request treats an existing output."""

    ENSURE_CURRENT = "ensure_current"
    CREATE_ONLY = "create_only"
    FORCE = "force"


class PartitionMaterializationRequest(BaseModel):
    """Logical UTC partitions requested for one materialization job."""

    model_config = ConfigDict(extra="forbid")

    timestamps: list[datetime] = Field(
        min_length=1,
        description="UTC timestamps aligned exactly to the start of an hour.",
        examples=[["2026-10-20T18:00:00Z"]],
    )

    @field_validator("timestamps")
    @classmethod
    def require_hour_aligned_utc_timestamps(
        cls, timestamps: list[datetime]
    ) -> list[datetime]:
        """Require every requested partition timestamp to be an exact UTC hour."""
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("Partition timestamps must be unique.")
        for timestamp in timestamps:
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("Partition timestamps must include a UTC offset.")
            if timestamp.utcoffset() != timedelta(0):
                raise ValueError("Partition timestamps must be in UTC.")
            if any((timestamp.minute, timestamp.second, timestamp.microsecond)):
                raise ValueError("Partition timestamps must be aligned to an exact hour.")
        return timestamps


class FeaturePartitionMaterializationRequest(PartitionMaterializationRequest):
    """Feature request with upstream-aware idempotency behavior."""

    mode: FeatureMaterializationMode = FeatureMaterializationMode.ENSURE_CURRENT


class JobResponse(BaseModel):
    """A materialization job and its aggregate lifecycle state."""

    id: UUID
    type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime


class PartitionUpdateRequestResponse(BaseModel):
    """A pending or terminal dataset or feature partition update request."""

    partition_id: UUID
    job_id: UUID
    timestamp: datetime = Field(
        description="UTC timestamp aligned exactly to the start of an hour.",
        examples=["2026-10-20T18:00:00Z"],
    )
    status: PartitionStatus
    created_at: datetime
    updated_at: datetime


class MaterializationJobResponse(BaseModel):
    """A newly created job with all of its requested partitions."""

    job: JobResponse | None
    partitions: list[PartitionUpdateRequestResponse]


class PartitionCompletionRequest(BaseModel):
    """A Spark worker's receipt for a committed Iceberg partition write."""

    model_config = ConfigDict(extra="forbid")

    content_hash: str = Field(min_length=1)
    snapshot_id: int


class FeaturePartitionCompletionRequest(PartitionCompletionRequest):
    """A feature callback confirms the immutable dataset revision plan it used."""

    dataset_partition_ids: list[UUID] = Field(min_length=1)


class PartitionResultResponse(BaseModel):
    """The recorded receipt for one successfully materialized partition."""

    partition_id: UUID
    content_hash: str
    snapshot_id: int
    created_at: datetime


class ErrorResponse(BaseModel):
    """Structured error response returned by the control plane."""

    detail: str


class ValidationErrorDetail(BaseModel):
    """One field-level request validation failure."""

    loc: list[str | int]
    msg: str
    type: str
    input: Any
    ctx: dict[str, Any] | None = None


class ValidationErrorResponse(BaseModel):
    """Structured request validation response returned with HTTP 422."""

    detail: list[ValidationErrorDetail]
