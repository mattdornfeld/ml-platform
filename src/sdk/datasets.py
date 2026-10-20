"""Typed runtime datasets injected into feature transformations."""

from dataclasses import dataclass

from pyspark.sql import DataFrame

from sdk.configs import DatasetMetadata


@dataclass(frozen=True)
class Dataset:
    """A declared dataset together with its dependency-window Spark DataFrame."""

    metadata: DatasetMetadata
    frame: DataFrame
