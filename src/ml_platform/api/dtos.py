"""HTTP request and response models for the materialization control plane."""

from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, computed_field, field_validator, model_validator

from ml_platform.common.types import BaseConfig


class _ApiModel(BaseConfig):
    """HTTP DTO base retaining JSON coercion while inheriting common safeguards."""

    model_config = ConfigDict(strict=False, frozen=False)


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


class PartitionStatus(StrEnum):
    """Lifecycle states for an individual partition update request."""

    PENDING = "pending"
    CREATED = "created"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeatureKey(StrEnum):
    """Supported logical keys for feature output rows."""

    SYMBOL = "symbol"


class DataValidations(_ApiModel):
    """Validation policy shared by every materialized resource."""

    allow_empty_partitions: bool = Field(
        default=False,
        description="Whether requested partitions may complete without emitted rows.",
    )


class FeatureValidations(DataValidations):
    """Configurable validation requirements for returned feature DataFrames."""

    min_rows_per_partition: int = Field(default=1, ge=1)
    require_finite_feature_value: bool = True
    require_feature_keys_from_input: bool = True


class DatasetType(StrEnum):
    """Dataset types accepted by the control plane."""

    MARKET_DATA = "market_data"


class DatasetTimeRange(_ApiModel):
    """Inclusive start and exclusive end dates for a dataset definition."""

    start_date: date
    end_date: date


class MarketDataDatasetConfig(_ApiModel):
    """Definition fields for a market-data dataset, excluding its derived ID."""

    description: str = Field(default="Market data.", min_length=1)
    validations: DataValidations = Field(default_factory=DataValidations)
    keys: list[FeatureKey] = Field(min_length=1)
    partition_type: Literal["hourly", "daily"]
    symbols: list[str] = Field(min_length=1)
    time_range: DatasetTimeRange


class DatasetCreateRequest(_ApiModel):
    """Payload for registering an immutable dataset definition."""

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    dataset_type: DatasetType
    config: MarketDataDatasetConfig

    @computed_field
    @property
    def id(self) -> str:
        """Return the immutable resource ID derived from name and version."""
        return f"{self.name}_v{self.version}"


class DatasetResponse(DatasetCreateRequest):
    """A persisted dataset definition."""

    created_at: datetime
    job_id: UUID | None = None


class DatasetDescriptionUpdate(_ApiModel):
    """Mutable display metadata for an existing dataset definition."""

    description: str = Field(min_length=1)


class FeatureDatasetDependency(_ApiModel):
    """A registered dataset version consumed by a feature."""

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    argument_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    num_lookback_partitions: int = Field(ge=0)


class FeatureCreateRequest(_ApiModel):
    """Payload for registering an immutable, normalized feature definition."""

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    dtype: Literal["double", "float", "int", "bigint"]
    frequency: Literal["hourly", "daily"]
    feature_keys: list[FeatureKey] = Field(default_factory=lambda: [FeatureKey.SYMBOL], min_length=1)
    datasets: list[FeatureDatasetDependency] = Field(min_length=1)
    transform: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    validations: FeatureValidations = Field(default_factory=FeatureValidations)

    @model_validator(mode="after")
    def validate_dataset_arguments(self) -> "FeatureCreateRequest":
        """Keep every dataset identity and transform argument unambiguous."""
        identities = {(dataset.name, dataset.version) for dataset in self.datasets}
        arguments = {dataset.argument_name for dataset in self.datasets}
        if len(identities) != len(self.datasets):
            raise ValueError("Feature datasets must have unique name and version identities.")
        if len(arguments) != len(self.datasets):
            raise ValueError("Feature datasets must have unique transform argument names.")
        if len(set(self.feature_keys)) != len(self.feature_keys):
            raise ValueError("Feature keys must be unique.")
        return self


class FeatureResponse(FeatureCreateRequest):
    """A persisted feature definition."""

    created_at: datetime


class FeatureMaterializationMode(StrEnum):
    """How a feature materialization request treats an existing output."""

    ENSURE_CURRENT = "ensure_current"
    CREATE_ONLY = "create_only"
    FORCE = "force"


class PartitionMaterializationRequest(_ApiModel):
    """Logical UTC partitions requested for one materialization job."""

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


class JobTarget(_ApiModel):
    """A versioned resource targeted by a materialization job."""

    resource_type: Literal["dataset", "feature", "label"]
    name: str
    version: int


class JobPartition(_ApiModel):
    """One resource partition targeted by a materialization job."""

    resource_type: Literal["dataset", "feature", "label"]
    name: str
    version: int
    timestamp: datetime
    status: PartitionStatus


class KubernetesResourceLink(_ApiModel):
    """A read-only Headlamp link for a Kubernetes resource created by a job."""

    resource_type: str
    resource_id: str
    url: str


class JobResponse(_ApiModel):
    """A materialization job, its targets, partitions, and aggregate lifecycle state."""

    id: UUID
    type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    targets: list[JobTarget] = Field(default_factory=list)
    partitions: list[JobPartition] = Field(default_factory=list)
    spark_application_name: str | None = None
    headlamp_url: str | None = None
    kubernetes_resources: list[KubernetesResourceLink] = Field(default_factory=list)


class PartitionUpdateRequestResponse(_ApiModel):
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


class MaterializationJobResponse(_ApiModel):
    """A newly created job with all of its requested partitions."""

    job: JobResponse | None
    partitions: list[PartitionUpdateRequestResponse]


class PartitionCompletionRequest(_ApiModel):
    """One authenticated receipt in an atomic Iceberg completion batch."""

    partition_id: UUID
    resource_type: Literal["dataset", "feature", "label"]
    row_count: int = Field(ge=0)
    schema_: dict[str, str] = Field(alias="schema", min_length=1)
    content_hash: str = Field(min_length=1)
    snapshot_id: int = Field(ge=0)
    callback_token: str = Field(min_length=1)
    dataset_partition_ids: list[UUID] | None = None

    @field_validator("schema_")
    @classmethod
    def require_named_schema_fields(cls, schema: dict[str, str]) -> dict[str, str]:
        """Reject empty column names or type declarations in committed schemas."""
        if any(not name.strip() or not dtype.strip() for name, dtype in schema.items()):
            raise ValueError("Completion schema requires non-empty column names and types.")
        return schema

    @model_validator(mode="after")
    def validate_dependencies(self) -> "PartitionCompletionRequest":
        """Derived resources must report their exact pinned dataset revisions."""
        if self.resource_type == "dataset" and self.dataset_partition_ids is not None:
            raise ValueError("Dataset receipts must not include dataset_partition_ids.")
        if self.resource_type != "dataset":
            if not self.dataset_partition_ids:
                raise ValueError("Derived receipts require dataset_partition_ids.")
            if len(set(self.dataset_partition_ids)) != len(self.dataset_partition_ids):
                raise ValueError("dataset_partition_ids must be unique.")
        return self


class PartitionResultResponse(_ApiModel):
    """The recorded receipt for one successfully materialized partition."""

    partition_id: UUID
    resource_type: Literal["dataset", "feature", "label"]
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")
    content_hash: str
    snapshot_id: int
    created_at: datetime


class ErrorResponse(_ApiModel):
    """Structured error response returned by the control plane."""

    detail: str


class ValidationErrorDetail(_ApiModel):
    """One field-level request validation failure."""

    loc: list[str | int]
    msg: str
    type: str
    input: Any
    ctx: dict[str, Any] | None = None


class ValidationErrorResponse(_ApiModel):
    """Structured request validation response returned with HTTP 422."""

    detail: list[ValidationErrorDetail]
