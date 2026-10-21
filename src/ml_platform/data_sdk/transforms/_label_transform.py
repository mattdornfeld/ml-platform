"""Base interface for Spark label transformations."""

from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class LabelTransform(ABC):
    """A configured transformation from one injected dataset to label rows."""

    @abstractmethod
    def __call__(self) -> DataFrame:
        """Return labels computed from datasets injected at construction."""
