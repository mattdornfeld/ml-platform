"""Feature API request and response DTOs."""

from enum import StrEnum

from pydantic import Field, model_validator

from ml_platform.common.api.dtos._base import _ApiModel
from ml_platform.common.api.dtos.partitions import (
    PartitionMaterializationRequest,
)
from ml_platform.common.types import (
    CreatedAt,
    DatasetName,
    DatasetType,
    DataType,
    Description,
    DiscriminatedValidationConfig,
    EntityKey,
    FeatureId,
    FeatureName,
    LookbackPartition,
    ParameterKey,
    ParameterValue,
    ResourceVersion,
    RunFrequency,
    TransformName,
)


class FeatureInputWindow(_ApiModel):
    """Historical source partitions available to one point-in-time feature."""

    lookback_partitions: LookbackPartition = Field(
        ge=0,
        description="Number of preceding source partitions required per output partition.",
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


class FeatureMaterializationMode(StrEnum):
    """How a feature materialization request treats an existing output."""

    ENSURE_CURRENT = "ensure_current"
    CREATE_ONLY = "create_only"
    FORCE = "force"


class FeaturePartitionMaterializationRequest(PartitionMaterializationRequest):
    """Feature request with upstream-aware idempotency behavior."""

    mode: FeatureMaterializationMode = Field(
        default=FeatureMaterializationMode.ENSURE_CURRENT,
        description="How this request handles partitions that already exist.",
    )
