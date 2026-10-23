"""HTTP request and response models for the materialization control plane."""

from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk.configs import (
    DataType,
    EntityKey,
    ParameterKey,
    ParameterValue,
    RunFrequency,
    TransformName,
)
from ml_platform.data_sdk.materializers._materialization_validator import (
    DiscriminatedValidationConfig,
)


class _ApiModel(BaseConfig):
    """HTTP DTO base retaining JSON coercion while inheriting common safeguards."""

    model_config = ConfigDict(
        strict=False,
        frozen=False,
        validate_by_name=True,
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


class PartitionStatus(StrEnum):
    """Lifecycle states for an individual partition update request."""

    PENDING = "pending"
    CREATED = "created"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetType(StrEnum):
    """Dataset types accepted by the control plane."""

    MARKET_DATA = "market_data"


class ResourceType(StrEnum):
    """Materializable resource kinds exposed by the control plane."""

    DATASET = "dataset"
    FEATURE = "feature"
    LABEL = "label"


class DatasetTimeRange(_ApiModel):
    """Inclusive start and exclusive end dates for a dataset definition."""

    start_date: date
    end_date: date


class MarketDataDatasetConfig(_ApiModel):
    """Definition fields for a market-data dataset, excluding its derived ID."""

    description: str = Field(default="Market data.", min_length=1)
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list
    )
    keys: list[EntityKey] = Field(min_length=1)
    partition_type: Literal["hourly", "daily"]
    symbols: list[str] = Field(min_length=1)
    time_range: DatasetTimeRange


class DatasetCreateRequest(_ApiModel):
    """Payload for registering an immutable dataset definition."""

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    dataset_type: DatasetType
    config: MarketDataDatasetConfig

    @computed_field  # type: ignore[prop-decorator]
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


class FeatureInputWindow(_ApiModel):
    """Historical source partitions available to one point-in-time feature."""

    lookback_partitions: int = Field(ge=0)


class LabelInputWindow(FeatureInputWindow):
    """Historical and realized-outcome source partitions required by a label."""

    lookahead_partitions: int = Field(ge=0)


class FeatureDatasetDependency(_ApiModel):
    """A point-in-time dataset version consumed by a feature."""

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    dataset_type: DatasetType
    input_window: FeatureInputWindow


class LabelDatasetDependency(FeatureDatasetDependency):
    """A realized-outcome dataset version consumed by a label."""

    input_window: LabelInputWindow


class FeatureCreateRequest(_ApiModel):
    """Payload for registering an immutable, normalized feature definition."""

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    dtype: DataType
    frequency: RunFrequency
    keys: list[EntityKey] = Field(
        default_factory=lambda: [EntityKey.SYMBOL], min_length=1
    )
    datasets: list[FeatureDatasetDependency] = Field(min_length=1)
    transform: TransformName = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[ParameterKey, ParameterValue] = Field(default_factory=dict)
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "FeatureCreateRequest":
        """Keep every dataset identity and transform input unambiguous."""
        identities = {
            (dataset.name, dataset.version) for dataset in self.datasets
        }
        if len(identities) != len(self.datasets):
            raise ValueError(
                "Feature datasets must have unique name and version identities."
            )
        if len({dataset.name for dataset in self.datasets}) != len(
            self.datasets
        ):
            raise ValueError(
                "Feature datasets must have unique names because dataset names "
                "are transform constructor arguments."
            )
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Entity keys must be unique.")
        return self


class FeatureResponse(FeatureCreateRequest):
    """A persisted feature definition."""

    created_at: datetime


class LabelCreateRequest(_ApiModel):
    """Payload for registering an immutable, normalized label definition."""

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    dtype: DataType
    frequency: RunFrequency
    keys: list[EntityKey] = Field(
        default_factory=lambda: [EntityKey.SYMBOL], min_length=1
    )
    datasets: list[LabelDatasetDependency] = Field(min_length=1)
    transform: TransformName = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[ParameterKey, ParameterValue] = Field(default_factory=dict)
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "LabelCreateRequest":
        """Keep every label input identity and transform input unambiguous."""
        identities = {
            (dataset.name, dataset.version) for dataset in self.datasets
        }
        if len(identities) != len(self.datasets):
            raise ValueError(
                "Label datasets must have unique name and version identities."
            )
        if len({dataset.name for dataset in self.datasets}) != len(
            self.datasets
        ):
            raise ValueError(
                "Label datasets must have unique names because dataset names "
                "are transform constructor arguments."
            )
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Entity keys must be unique.")
        return self


class LabelResponse(LabelCreateRequest):
    """A persisted label definition."""

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


class FeaturePartitionMaterializationRequest(PartitionMaterializationRequest):
    """Feature request with upstream-aware idempotency behavior."""

    mode: FeatureMaterializationMode = FeatureMaterializationMode.ENSURE_CURRENT


class LabelPartitionMaterializationRequest(
    FeaturePartitionMaterializationRequest
):
    """Label request with upstream-aware idempotency behavior."""


class JobTarget(_ApiModel):
    """A versioned resource targeted by a materialization job."""

    resource_type: ResourceType
    name: str
    version: int


class JobPartition(_ApiModel):
    """One resource partition targeted by a materialization job."""

    resource_type: ResourceType
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
    kubernetes_resources: list[KubernetesResourceLink] = Field(
        default_factory=list
    )


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
    resource_type: ResourceType
    row_count: int = Field(ge=0)
    schema_: dict[str, str] = Field(alias="schema", min_length=1)
    content_hash: str = Field(min_length=1)
    snapshot_id: int = Field(ge=0)
    callback_token: str = Field(min_length=1)
    dataset_partition_ids: list[UUID] | None = None

    @field_validator("schema_")
    @classmethod
    def require_named_schema_fields(
        cls, schema: dict[str, str]
    ) -> dict[str, str]:
        """Reject empty column names or type declarations in committed schemas."""
        if any(
            not name.strip() or not dtype.strip()
            for name, dtype in schema.items()
        ):
            raise ValueError(
                "Completion schema requires non-empty column names and types."
            )
        return schema

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


class PartitionResultResponse(_ApiModel):
    """The recorded receipt for one successfully materialized partition."""

    partition_id: UUID
    resource_type: ResourceType
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
