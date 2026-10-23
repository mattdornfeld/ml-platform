"""Common immutable resource metadata models."""

from collections.abc import Sequence
from typing import Self

from pydantic import Field, field_validator, model_validator

from ml_platform.common.domain._base import DomainConfig
from ml_platform.common.domain._exceptions import (
    TransformDependencyConfigurationError,
)
from ml_platform.common.domain.dependencies import DatasetDependency
from ml_platform.common.types import (
    DataType,
    Description,
    DiscriminatedValidationConfig,
    EntityKey,
    ParameterKey,
    ParameterValue,
    PartitionType,
    ResourceVersion,
    RunFrequency,
    TransformName,
)


class DataConfig[NameT: str](DomainConfig):
    """Common immutable metadata for every versioned resource definition."""

    name: NameT = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase resource name.",
    )
    version: ResourceVersion = Field(
        ge=1, description="Positive immutable resource-definition version."
    )
    description: Description = Field(
        default=Description("No description provided."),
        min_length=1,
        description="Human-readable resource description.",
    )
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Data quality validations applied during materialization.",
    )


class TransformConfig[NameT: str](DataConfig[NameT]):
    """Common typed transformation definition shared by features and labels."""

    dtype: DataType = Field(
        description="Spark SQL type used to persist the transform value."
    )
    frequency: RunFrequency = Field(
        description="Cadence of the transform's materialized partitions."
    )
    keys: list[EntityKey] = Field(
        default_factory=lambda: [EntityKey.SYMBOL],
        min_length=1,
        description="Ordered entity-key columns that identify each transform row.",
    )
    datasets: Sequence[DatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into transforms by dataset name.",
    )
    transform: TransformName = Field(
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
        description="Absolute import path of the transform implementation.",
    )
    parameters: dict[ParameterKey, ParameterValue] = Field(
        default_factory=dict,
        description="Typed keyword arguments passed to the transform constructor.",
    )

    @property
    def partition_type(self) -> PartitionType:
        """Return the physical cadence derived from transform frequency."""
        return PartitionType(self.frequency)

    @field_validator("description", "transform")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only descriptions and transform paths."""
        if not value.strip():
            raise TransformDependencyConfigurationError(
                "Transform text fields must contain non-whitespace text."
            )
        return value

    @field_validator("parameters")
    @classmethod
    def require_non_blank_parameter_keys(
        cls, parameters: dict[ParameterKey, ParameterValue]
    ) -> dict[ParameterKey, ParameterValue]:
        """Reject whitespace-only transform parameter names."""
        if any(not parameter_key.strip() for parameter_key in parameters):
            raise TransformDependencyConfigurationError(
                "Transform parameter keys must not be blank."
            )
        return parameters

    @field_validator("keys", mode="before")
    @classmethod
    def parse_keys(cls, keys: list[EntityKey | str]) -> list[EntityKey]:
        """Parse YAML key literals into the closed EntityKey enum."""
        return [EntityKey(key) for key in keys]

    @model_validator(mode="after")
    def validate_transform_dependencies(self) -> Self:
        """Require unambiguous transform identities and injected datasets."""
        if len({dataset.id for dataset in self.datasets}) != len(self.datasets):
            raise TransformDependencyConfigurationError(
                "Transform datasets must have unique name/version identities."
            )
        if len({dataset.name for dataset in self.datasets}) != len(
            self.datasets
        ):
            raise TransformDependencyConfigurationError(
                "Transform datasets must have unique names because dataset names "
                "are transform constructor arguments."
            )
        if len(set(self.keys)) != len(self.keys):
            raise TransformDependencyConfigurationError(
                "Transform entity keys must be unique."
            )
        validation_types = {validation.type for validation in self.validations}
        if len(validation_types) != len(self.validations):
            raise TransformDependencyConfigurationError(
                "Transform validation types must be unique."
            )
        return self
