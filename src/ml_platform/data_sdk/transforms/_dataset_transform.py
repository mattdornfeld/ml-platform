"""Base interface for transforms that construct partition-targeted dataset DataFrames."""

from abc import ABC, abstractmethod

from pyspark.sql import SparkSession

from ml_platform.data_sdk.transforms._base import DataTransform
from ml_platform.data_sdk.transforms._dataset_schema import (
    DatasetPartitions,
    DatasetSchema,
    EntityValueSet,
)
from ml_platform.data_sdk.utils import record


@record
class DatasetTransform(DataTransform, ABC):
    """A transform that constructs rows for exactly the requested partitions."""

    spark: SparkSession
    partitions: DatasetPartitions

    @property
    @abstractmethod
    def schema(self) -> DatasetSchema:
        """Return the statically declared output schema."""

    @property
    @abstractmethod
    def entity_value_sets(self) -> tuple[EntityValueSet, ...]:
        """Return configured entity values for synthetic generation."""
