"""Shared strict Pydantic types."""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Immutable, strict base for ML Platform Pydantic configuration models."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        revalidate_instances="always",
        arbitrary_types_allowed=False,
        allow_inf_nan=False,
    )
