"""Base interface for transforms that construct partition-targeted dataset DataFrames."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

from pyspark.sql import DataFrame


class DatasetTransform(ABC):
    """A transform that constructs rows for exactly the injected logical partitions."""

    partitions: Sequence[datetime]

    @abstractmethod
    def __call__(self) -> DataFrame:
        """Return only rows belonging to the injected logical partitions."""
