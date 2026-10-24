"""Private base and cross-module HTTP errors for the control-plane API."""

from fastapi import HTTPException, status


class _ApiError(HTTPException):
    """Base class for concrete control-plane HTTP errors."""


class DatabaseUrlNotConfiguredError(_ApiError):
    """The API lacks the configured PostgreSQL endpoint."""

    def __init__(self) -> None:
        """Build the database-configuration service-unavailable response."""
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DATABASE_URL must be configured for materialization services.",
        )


class InvalidRuntimeEnvironmentError(_ApiError):
    """The API received an unsupported ML Platform runtime environment."""

    def __init__(self, environment: str) -> None:
        """Build the invalid-environment service configuration response."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported ML Platform environment {environment!r}.",
        )


class UnsupportedScheduledJobTypeError(_ApiError):
    """A persisted job type has no scheduler implementation."""

    def __init__(self, job_type: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported scheduled job type {job_type!r}.",
        )
