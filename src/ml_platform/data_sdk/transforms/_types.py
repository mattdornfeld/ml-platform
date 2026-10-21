"""Private runtime types shared by data transformations and materializers."""

from dataclasses import dataclass

from pydantic import Field

from ml_platform.common.types import BaseConfig
from pyspark.sql import DataFrame


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
