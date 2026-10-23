"""HTTP errors raised while acquiring the materialization database."""

from fastapi import status

from ml_platform.api._exceptions import _ApiError


class DatabaseUrlNotConfiguredError(_ApiError):
    """The API lacks the configured PostgreSQL endpoint."""

    def __init__(self) -> None:
        """Build the database-configuration service-unavailable response."""
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DATABASE_URL must be configured for materialization services.",
        )


class MaterializationDatabaseUnavailableError(_ApiError):
    """PostgreSQL cannot currently serve materialization requests."""

    def __init__(self) -> None:
        """Build the database-availability service-unavailable response."""
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Materialization database is unavailable.",
        )
