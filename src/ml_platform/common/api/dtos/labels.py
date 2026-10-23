"""Label API request and response DTOs."""

from pydantic import Field, field_validator, model_validator

from ml_platform.common.api.dtos._base import (
    _ApiModel,
    _require_unique_validation_types,
)
from ml_platform.common.api.dtos.features import (
    FeatureDatasetDependency,
    FeatureInputWindow,
    FeaturePartitionMaterializationRequest,
)
from ml_platform.common.types import (
    CreatedAt,
    DataType,
    Description,
    DiscriminatedValidationConfig,
    EntityKey,
    LabelId,
    LabelName,
    LookAheadPartitions,
    ParameterKey,
    ParameterValue,
    ResourceVersion,
    RunFrequency,
    TransformName,
)


class LabelInputWindow(FeatureInputWindow):
    """Historical and realized-outcome source partitions required by a label."""

    lookahead_partitions: LookAheadPartitions = Field(
        ge=0,
        description="Number of following source partitions required per output partition.",
    )


class LabelDatasetDependency(FeatureDatasetDependency):
    """A realized-outcome dataset version consumed by a label."""

    input_window: LabelInputWindow = Field(
        description="Historical and forward source-partition range required by the label.",
    )


class LabelCreateRequest(_ApiModel):
    """Payload for registering an immutable, normalized label definition."""

    name: LabelName = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable label name.",
    )
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
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
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
        _require_unique_validation_types(self.validations)
        return self

    @field_validator("description", "transform")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only descriptive and import-path fields."""
        if not value.strip():
            raise ValueError("Value must contain non-whitespace text.")
        return value

    @field_validator("parameters")
    @classmethod
    def require_non_blank_parameter_keys(
        cls, parameters: dict[ParameterKey, ParameterValue]
    ) -> dict[ParameterKey, ParameterValue]:
        """Reject whitespace-only transform parameter names."""
        if any(not parameter_key.strip() for parameter_key in parameters):
            raise ValueError("Transform parameter keys must not be blank.")
        return parameters


class LabelResponse(LabelCreateRequest):
    """A persisted label definition."""

    created_at: CreatedAt = Field(
        description="UTC time at which the label definition was registered.",
    )


class LabelPartitionMaterializationRequest(
    FeaturePartitionMaterializationRequest
):
    """Label request with upstream-aware idempotency behavior."""
