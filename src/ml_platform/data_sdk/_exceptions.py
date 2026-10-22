"""Shared exception roots for the public data SDK error hierarchy."""


class DataSdkError(ValueError):
    """Base exception for every expected ML Platform data-SDK failure."""


class MarketSymbolConfigurationError(DataSdkError):
    """Raised when a market-data definition contains no usable ticker symbols."""
