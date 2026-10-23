"""Dataset API request and response DTOs."""

from enum import StrEnum

from pydantic import Field, computed_field, field_validator, model_validator

from ml_platform.common.api.dtos._base import (
    _ApiModel,
    _require_unique_validation_types,
)
from ml_platform.common.types import (
    CreatedAt,
    DatasetId,
    DatasetName,
    DatasetType,
    Description,
    DiscriminatedValidationConfig,
    EntityKey,
    PartitionType,
    ResourceVersion,
    Symbol,
)


class DatasetStatus(StrEnum):
    """Lifecycle states accepted when listing registered datasets."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


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

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, symbols: list[Symbol]) -> list[Symbol]:
        """Normalize Yahoo Finance symbols and reject blank or duplicate values."""
        normalized_symbols = [
            Symbol(symbol.strip().upper()) for symbol in symbols
        ]
        if any(not symbol for symbol in normalized_symbols):
            raise ValueError("Market-data symbols must not be blank.")
        if len(set(normalized_symbols)) != len(normalized_symbols):
            raise ValueError("Market-data symbols must be unique.")
        return normalized_symbols

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, description: str) -> str:
        """Require a meaningful market-data dataset description."""
        if not description.strip():
            raise ValueError("Description must contain non-whitespace text.")
        return description

    @model_validator(mode="after")
    def validate_configuration(self) -> "MarketDataDatasetConfig":
        """Keep dataset entity keys and validation overrides unambiguous."""
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Dataset entity keys must be unique.")
        _require_unique_validation_types(self.validations)
        return self


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
