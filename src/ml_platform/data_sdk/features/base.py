"""Base interface and parameter validation for Spark feature transforms."""

from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class FeatureTransform(ABC):
    """A configured transformation from injected datasets to a feature DataFrame."""

    @abstractmethod
    def __call__(self) -> DataFrame:
        """Return ``timestamp``, ``symbol``, and ``feature_value`` columns."""
