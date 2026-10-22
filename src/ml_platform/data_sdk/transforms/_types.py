"""Private runtime types shared by data transformations and materializers."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from ml_platform.common.types import BaseConfig
from pyspark.sql import DataFrame


class FeatureKey(StrEnum):
    """Supported logical entity keys emitted by data transforms."""

    SYMBOL = "symbol"


class DataType(StrEnum):
    """Supported Iceberg value types emitted by feature and label transforms."""

    DOUBLE = "double"
    FLOAT = "float"
    INT = "int"
    BIGINT = "bigint"


class PartitionType(StrEnum):
    """Supported Iceberg timestamp partition transforms."""

    HOURLY = "hourly"
    DAILY = "daily"


class DatasetDependency(BaseConfig):
    """Immutable declared dataset dependency for a derived materialization."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Stable lowercase name of the upstream dataset.")
    version: int = Field(ge=1, description="Immutable version of the upstream dataset.")
    num_lookback_partitions: int = Field(
        ge=0, description="Number of preceding dataset partitions required for each output partition."
    )

    @property
    def id(self) -> str:
        """Return the canonical resource ID."""
        return f"{self.name}_v{self.version}"


class FeatureDatasetDependency(DatasetDependency):
    """A declared dataset mapped to one feature transform constructor argument."""

    argument_name: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Feature transform constructor argument receiving this dataset.",
    )


@dataclass(frozen=True)
class DatasetReference:
    """Immutable dataset identity and dependency range injected into a transform."""

    name: str
    version: int
    num_lookback_partitions: int

    @property
    def id(self) -> str:
        """Return the canonical resource ID."""
        return f"{self.name}_v{self.version}"


@dataclass(frozen=True)
class Dataset:
    """A dataset reference together with its dependency-window DataFrame."""

    metadata: DatasetReference
    frame: DataFrame
