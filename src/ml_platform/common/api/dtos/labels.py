"""Label API request and response DTOs."""

from pydantic import Field

from ml_platform.common.api.dtos.features import (
    FeaturePartitionMaterializationRequest,
)
from ml_platform.common.domain import (
    LabelConfig,
    LabelDatasetDependency,
    LabelInputWindow,
)
from ml_platform.common.types import CreatedAt


class LabelCreateRequest(LabelConfig):
    """Payload for registering an immutable normalized label definition."""


class LabelResponse(LabelCreateRequest):
    """A persisted label definition."""

    created_at: CreatedAt = Field(
        description="UTC time at which the label definition was registered.",
    )


class LabelPartitionMaterializationRequest(
    FeaturePartitionMaterializationRequest
):
    """Label request with upstream-aware idempotency behavior."""


__all__ = [
    "LabelCreateRequest",
    "LabelDatasetDependency",
    "LabelInputWindow",
    "LabelPartitionMaterializationRequest",
    "LabelResponse",
]
