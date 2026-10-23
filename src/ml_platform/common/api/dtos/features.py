"""Feature API request and response DTOs."""

from enum import StrEnum

from pydantic import Field

from ml_platform.common.api.dtos.partitions import (
    PartitionMaterializationRequest,
)
from ml_platform.common.domain import (
    FeatureConfig,
    FeatureDatasetDependency,
    FeatureInputWindow,
)
from ml_platform.common.types import CreatedAt


class FeatureCreateRequest(FeatureConfig):
    """Payload for registering an immutable normalized feature definition."""


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


__all__ = [
    "FeatureCreateRequest",
    "FeatureDatasetDependency",
    "FeatureInputWindow",
    "FeatureMaterializationMode",
    "FeaturePartitionMaterializationRequest",
    "FeatureResponse",
]
