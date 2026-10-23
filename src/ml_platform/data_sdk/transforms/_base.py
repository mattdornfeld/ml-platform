"""Shared output table contract for dataset, feature, and label transforms."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pyspark.sql import DataFrame

from ml_platform.common.types import EntityKey, PartitionType
from ml_platform.data_sdk._exceptions import DataSdkError
from ml_platform.data_sdk.utils import record


class DataTransformSchemaError(DataSdkError):
    """Raised when a transform's key and value schemas overlap."""


@record
class DataTransform(ABC):
    """Shared base for user-defined dataset, feature, and label calculations."""

    keys: Sequence[EntityKey]
    partition_type: PartitionType

    @property
    def key_schema(self) -> dict[str, str]:
        return {
            "timestamp": "timestamp",
            **{key.value: "string" for key in self.keys},
        }

    @property
    @abstractmethod
    def value_schema(self) -> dict[str, str]: ...

    @property
    def iceberg_schema(self) -> str:
        """Return the Spark SQL declaration assembled from key and value schemas."""
        key_schema = self.key_schema
        value_schema = self.value_schema
        overlapping_columns = key_schema.keys() & value_schema.keys()
        if overlapping_columns:
            raise DataTransformSchemaError(
                "Key and value schemas must not share columns: "
                f"{', '.join(sorted(overlapping_columns))}."
            )
        columns = key_schema | value_schema
        return ", ".join(
            f"{name} {data_type.upper()}" for name, data_type in columns.items()
        )

    @property
    def iceberg_partition_spec(self) -> str:
        """Return the Iceberg timestamp transform for the static partition type."""
        match self.partition_type:
            case PartitionType.HOURLY:
                return "hours(timestamp)"
            case PartitionType.DAILY:
                return "days(timestamp)"

    @abstractmethod
    def __call__(self) -> DataFrame: ...
