"""Dataset API request and response DTOs."""

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from ml_platform.common.api.dtos._base import _ApiModel
from ml_platform.common.domain import (
    DatasetTypeConfigurationError,
    MarketDataDatasetConfig,
)
from ml_platform.common.types import (
    CreatedAt,
    DatasetId,
    DatasetName,
    DatasetType,
    Description,
    ResourceVersion,
)


class DatasetStatus(StrEnum):
    """Lifecycle states accepted when listing registered datasets."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class DatasetCreateRequest(_ApiModel):
    """Payload for registering an immutable dataset definition."""

    name: DatasetName = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
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

    @model_validator(mode="after")
    def validate_dataset_type(self) -> "DatasetCreateRequest":
        """Require the envelope and nested configuration to name one type."""
        if self.dataset_type is not self.config.dataset_type:
            raise DatasetTypeConfigurationError(
                "Dataset type must match the nested dataset configuration."
            )
        return self


class DatasetResponse(DatasetCreateRequest):
    """A persisted dataset definition."""

    id: DatasetId = Field(
        description="Immutable ID derived from the persisted name and version."
    )
    created_at: CreatedAt = Field(
        description="UTC time at which the dataset definition was registered.",
    )


class DatasetDescriptionUpdate(_ApiModel):
    """Mutable display metadata for an existing dataset definition."""

    description: Description = Field(
        min_length=1,
        description="Replacement human-readable dataset description.",
    )

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, description: str) -> str:
        """Require meaningful replacement description text."""
        if not description.strip():
            raise ValueError("Description must contain non-whitespace text.")
        return description
