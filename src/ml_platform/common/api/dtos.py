"""HTTP request and response models for the materialization control plane."""

from datetime import timedelta
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from ml_platform.common.types import (
    CallbackToken,
    ContentHash,
    CreatedAt,
    DatasetId,
    DatasetName,
    DatasetType,
    DataType,
    Description,
    DiscriminatedValidationConfig,
    EntityKey,
    FeatureId,
    FeatureName,
    IcebergSchema,
    JobId,
    LabelId,
    LabelName,
    LookAheadPartitions,
    LookbackPartition,
    ParameterKey,
    ParameterValue,
    PartitionId,
    PartitionTimestamp,
    PartitionType,
    ResourceName,
    ResourceVersion,
    RowCount,
    RunFrequency,
    SnapshotId,
    Symbol,
    TransformName,
    UpdatedAt,
)


class _ApiModel(BaseModel):
    """Frozen HTTP DTO base with only JSON-boundary type coercion enabled."""

    model_config = ConfigDict(
        extra="forbid",
        strict=False,
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        revalidate_instances="always",
        arbitrary_types_allowed=False,
        allow_inf_nan=False,
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


class ResourceType(StrEnum):
    """Materializable resource kinds exposed by the control plane."""

    DATASET = "dataset"
    FEATURE = "feature"
    LABEL = "label"


class MarketDataDatasetConfig(_ApiModel):
    """Definition fields for a market-data dataset, excluding its derived ID."""

    description: Description = Field(
        default=Description("Market data."),
        min_length=1,
        description="Human-readable explanation of the dataset.",
    )
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Validation policies applied when materializing the dataset.",
    )
    keys: list[EntityKey] = Field(
        min_length=1,
        description="Entity columns that identify each market-data observation.",
    )
    partition_type: PartitionType = Field(
        description="Iceberg timestamp partition cadence for the dataset.",
    )
    symbols: list[Symbol] = Field(
        min_length=1,
        description="Yahoo Finance ticker symbols to retrieve.",
    )


class DatasetCreateRequest(_ApiModel):
    """Payload for registering an immutable dataset definition."""

    name: DatasetName = Field(
        min_length=1,
        description="Stable dataset name shared by all of its versions.",
    )
    version: ResourceVersion = Field(
        gt=0,
        description="Positive immutable version number for this dataset definition.",
    )
    dataset_type: DatasetType = Field(
        description="Transform family that produces this dataset.",
    )
    config: MarketDataDatasetConfig = Field(
        description="Dataset-type-specific materialization definition.",
    )

    @computed_field(  # type: ignore[prop-decorator]
        description="Immutable ID derived from the name and version."
    )
    @property
    def id(self) -> DatasetId:
        """Return the immutable dataset ID derived from name and version."""
        return DatasetId(f"{self.name}_v{self.version}")


class DatasetResponse(DatasetCreateRequest):
    """A persisted dataset definition."""

    created_at: CreatedAt = Field(
        description="UTC time at which the dataset definition was registered.",
    )


class DatasetDescriptionUpdate(_ApiModel):
    """Mutable display metadata for an existing dataset definition."""

    description: Description = Field(
        min_length=1,
        description="Replacement human-readable dataset description.",
    )


class FeatureInputWindow(_ApiModel):
    """Historical source partitions available to one point-in-time feature."""

    lookback_partitions: LookbackPartition = Field(
        ge=0,
        description="Number of preceding source partitions required per output partition.",
    )


class LabelInputWindow(FeatureInputWindow):
    """Historical and realized-outcome source partitions required by a label."""

    lookahead_partitions: LookAheadPartitions = Field(
        ge=0,
        description="Number of following source partitions required per output partition.",
    )


class FeatureDatasetDependency(_ApiModel):
    """A point-in-time dataset version consumed by a feature."""

    name: DatasetName = Field(
        min_length=1,
        description="Name of the upstream dataset.",
    )
    version: ResourceVersion = Field(
        gt=0,
        description="Immutable version of the upstream dataset.",
    )
    dataset_type: DatasetType = Field(
        description="Transform family of the upstream dataset.",
    )
    input_window: FeatureInputWindow = Field(
        description="Historical source-partition range required by the transform.",
    )


class LabelDatasetDependency(FeatureDatasetDependency):
    """A realized-outcome dataset version consumed by a label."""

    input_window: LabelInputWindow = Field(
        description="Historical and forward source-partition range required by the label.",
    )


class FeatureCreateRequest(_ApiModel):
    """Payload for registering an immutable, normalized feature definition."""

    name: FeatureName = Field(min_length=1, description="Stable feature name.")
    version: ResourceVersion = Field(
        gt=0, description="Positive immutable feature version number."
    )
    dtype: DataType = Field(
        description="Persisted Spark type for feature values."
    )
    frequency: RunFrequency = Field(
        description="Cadence at which feature partitions are materialized.",
    )
    keys: list[EntityKey] = Field(
        default_factory=lambda: [EntityKey.SYMBOL],
        min_length=1,
        description="Entity columns that uniquely identify each feature value.",
    )
    datasets: list[FeatureDatasetDependency] = Field(
        min_length=1,
        description="Versioned upstream datasets injected into the transform.",
    )
    transform: TransformName = Field(
        min_length=1,
        description="Import path of the feature transform class.",
    )
    description: Description = Field(
        min_length=1, description="Human-readable explanation of the feature."
    )
    parameters: dict[ParameterKey, ParameterValue] = Field(
        default_factory=dict,
        description="Typed constructor parameters passed to the feature transform.",
    )
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Validation policies applied when materializing the feature.",
    )

    @property
    def id(self) -> FeatureId:
        """Return the immutable feature ID derived from name and version."""
        return FeatureId(f"{self.name}_v{self.version}")

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

    created_at: CreatedAt = Field(
        description="UTC time at which the feature definition was registered.",
    )


class LabelCreateRequest(_ApiModel):
    """Payload for registering an immutable, normalized label definition."""

    name: LabelName = Field(min_length=1, description="Stable label name.")
    version: ResourceVersion = Field(
        gt=0, description="Positive immutable label version number."
    )
    dtype: DataType = Field(
        description="Persisted Spark type for label values."
    )
    frequency: RunFrequency = Field(
        description="Cadence at which label partitions are materialized.",
    )
    keys: list[EntityKey] = Field(
        default_factory=lambda: [EntityKey.SYMBOL],
        min_length=1,
        description="Entity columns that uniquely identify each label value.",
    )
    datasets: list[LabelDatasetDependency] = Field(
        min_length=1,
        description="Versioned upstream datasets injected into the transform.",
    )
    transform: TransformName = Field(
        min_length=1,
        description="Import path of the label transform class.",
    )
    description: Description = Field(
        min_length=1, description="Human-readable explanation of the label."
    )
    parameters: dict[ParameterKey, ParameterValue] = Field(
        default_factory=dict,
        description="Typed constructor parameters passed to the label transform.",
    )
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Validation policies applied when materializing the label.",
    )

    @property
    def id(self) -> LabelId:
        """Return the immutable label ID derived from name and version."""
        return LabelId(f"{self.name}_v{self.version}")

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

    created_at: CreatedAt = Field(
        description="UTC time at which the label definition was registered.",
    )


class FeatureMaterializationMode(StrEnum):
    """How a feature materialization request treats an existing output."""

    ENSURE_CURRENT = "ensure_current"
    CREATE_ONLY = "create_only"
    FORCE = "force"


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


class FeaturePartitionMaterializationRequest(PartitionMaterializationRequest):
    """Feature request with upstream-aware idempotency behavior."""

    mode: FeatureMaterializationMode = Field(
        default=FeatureMaterializationMode.ENSURE_CURRENT,
        description="How this request handles partitions that already exist.",
    )


class LabelPartitionMaterializationRequest(
    FeaturePartitionMaterializationRequest
):
    """Label request with upstream-aware idempotency behavior."""


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


class ErrorResponse(_ApiModel):
    """Structured error response returned by the control plane."""

    detail: str = Field(description="Human-readable explanation of the error.")


class ValidationErrorDetail(_ApiModel):
    """One field-level request validation failure."""

    loc: list[str | int] = Field(
        description="Location of the invalid value within the request body.",
    )
    msg: str = Field(description="Human-readable validation failure message.")
    type: str = Field(description="Pydantic validation error identifier.")
    input: Any = Field(description="Input value that failed validation.")
    ctx: dict[str, Any] | None = Field(
        default=None,
        description="Additional context supplied by the validation failure.",
    )


class ValidationErrorResponse(_ApiModel):
    """Structured request validation response returned with HTTP 422."""

    detail: list[ValidationErrorDetail] = Field(
        description="Field-level validation failures in the rejected request.",
    )
