"""Standard control-plane error response DTOs."""

from typing import Any

from pydantic import Field

from ml_platform.common.api.dtos._base import _ApiModel


class ErrorResponse(_ApiModel):
    """Structured error response returned by the control plane."""

    detail: str = Field(description="Human-readable explanation of the error.")


class ValidationErrorDetail(_ApiModel):
    """One field-level request validation failure."""

    loc: list[str | int] = Field(
        description="Location of the invalid value within the request body.",
    )
    msg: str = Field(description="Human-readable validation failure message.")
    type: str = Field(description="Pydantic validation error identifier.")
    input: Any = Field(description="Input value that failed validation.")
    ctx: dict[str, Any] | None = Field(
        default=None,
        description="Additional context supplied by the validation failure.",
    )


class ValidationErrorResponse(_ApiModel):
    """Structured request validation response returned with HTTP 422."""

    detail: list[ValidationErrorDetail] = Field(
        description="Field-level validation failures in the rejected request.",
    )
