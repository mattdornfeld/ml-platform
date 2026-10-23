"""Typed HTTP errors raised directly by control-plane route handlers."""

from fastapi import status

from ml_platform.api._exceptions import _ApiError


class DatasetVersionNotFoundError(_ApiError):
    """The requested dataset version is absent."""

    def __init__(self) -> None:
        """Initialize this fixed HTTP error response."""
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset version does not exist.",
        )


class FeatureVersionNotFoundError(_ApiError):
    """The requested feature version is absent."""

    def __init__(self) -> None:
        """Initialize this fixed HTTP error response."""
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature version does not exist.",
        )


class LabelVersionNotFoundError(_ApiError):
    """The requested label version is absent."""

    def __init__(self) -> None:
        """Initialize this fixed HTTP error response."""
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Label version does not exist.",
        )


class MaterializationJobNotFoundError(_ApiError):
    """The requested materialization job is absent."""

    def __init__(self) -> None:
        """Initialize this fixed HTTP error response."""
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Materialization job does not exist.",
        )
