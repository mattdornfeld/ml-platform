"""Shared Pydantic configuration for control-plane HTTP DTOs."""

from pydantic import BaseModel, ConfigDict


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
