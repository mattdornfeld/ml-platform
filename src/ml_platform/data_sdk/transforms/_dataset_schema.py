"""Strongly typed synthetic-data schema declarations for dataset transforms."""

from datetime import datetime
from enum import StrEnum
from typing import NewType

from ml_platform.common.types import EntityKey, PartitionType, Symbol
from ml_platform.data_sdk.utils import record

DatasetColumnName = NewType("DatasetColumnName", str)
EntityValue = NewType("EntityValue", str)
type DatasetPartitions = tuple[datetime, ...]
type EntityValues = tuple[EntityValue, ...]
type MarketSymbols = tuple[Symbol, ...]
type IsNullable = bool
type SyntheticSeed = int
type DistributionValue = float
type DistributionMinimum = float
type DistributionMaximum = float
type DistributionMean = float
type DistributionStandardDeviation = float
type SyntheticConstant = str | int | float | bool


class DatasetColumnType(StrEnum):
    """Spark-compatible types emitted by dataset transforms."""

    TIMESTAMP = "timestamp"
    STRING = "string"
    DOUBLE = "double"
    BIGINT = "bigint"


@record
class ConstantDistribution:
    """Emit one deterministic constant for every generated row."""

    value: SyntheticConstant


@record
class UniformDistribution:
    """Sample a numeric value uniformly over an inclusive range."""

    minimum: DistributionMinimum
    maximum: DistributionMaximum


@record
class NormalDistribution:
    """Sample a numeric value from a normal distribution."""

    mean: DistributionMean
    standard_deviation: DistributionStandardDeviation


@record
class EntityValueDistribution:
    """Emit an entity value supplied by the configured transform."""

    key: EntityKey


@record
class PartitionTimestampDistribution:
    """Emit the timestamp of the partition being materialized."""


type SyntheticDistribution = (
    ConstantDistribution
    | UniformDistribution
    | NormalDistribution
    | EntityValueDistribution
    | PartitionTimestampDistribution
)


@record
class EntityValueSet:
    """Configured values emitted for one entity-key column."""

    key: EntityKey
    values: EntityValues


@record
class DatasetColumn:
    """One output column with a type and deterministic sampling distribution."""

    name: DatasetColumnName
    column_type: DatasetColumnType
    nullable: IsNullable
    distribution: SyntheticDistribution


@record
class PositiveInvariant:
    """Require every value in one numeric column to be strictly positive."""

    column: DatasetColumnName


@record
class LessThanOrEqualInvariant:
    """Require every lower-column value to be at most its upper-column value."""

    lower_column: DatasetColumnName
    upper_column: DatasetColumnName


type SchemaInvariant = PositiveInvariant | LessThanOrEqualInvariant


@record
class DatasetSchema:
    """Static output contract for real and synthetic dataset transforms."""

    columns: tuple[DatasetColumn, ...]
    invariants: tuple[SchemaInvariant, ...]
    partition_type: PartitionType
