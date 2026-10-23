"""Shared Pydantic configuration and validators for HTTP DTOs."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from ml_platform.common.types import ValidationConfig


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


def _require_unique_validation_types(
    validations: Sequence[ValidationConfig],
) -> Sequence[ValidationConfig]:
    """Reject ambiguous overrides of one materialization validation type."""
    validation_types = {validation.type for validation in validations}
    if len(validation_types) != len(validations):
        raise ValueError("Validation policies must use unique types.")
    return validations
