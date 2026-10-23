"""Dataset API request and response DTOs."""

from enum import StrEnum

from pydantic import Field, computed_field, field_validator

from ml_platform.common.api.dtos._base import _ApiModel
from ml_platform.common.domain import MarketDataDatasetConfig
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

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, description: str) -> str:
        """Require meaningful replacement description text."""
        if not description.strip():
            raise ValueError("Description must contain non-whitespace text.")
        return description
