"""Private base and cross-module HTTP errors for the control-plane API."""

from fastapi import HTTPException, status


class _ApiError(HTTPException):
    """Base class for concrete control-plane HTTP errors."""


class UnsupportedScheduledJobTypeError(_ApiError):
    """A persisted job type has no scheduler implementation."""

    def __init__(self, job_type: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported scheduled job type {job_type!r}.",
        )
