"""Strongly typed synthetic-data schema declarations for dataset transforms."""

from datetime import datetime
from enum import StrEnum
from typing import NewType

from ml_platform.common.types import EntityKey, PartitionType, Symbol
from ml_platform.data_sdk.utils import record

#: Stable identifier for one declared dataset column.
DatasetColumnName = NewType("DatasetColumnName", str)
#: One configured concrete value for an entity-key column.
EntityValue = NewType("EntityValue", str)
#: Immutable requested materialization partition timestamps.
type DatasetPartitions = tuple[datetime, ...]
#: Immutable concrete values emitted for one entity key.
type EntityValues = tuple[EntityValue, ...]
#: Immutable Yahoo Finance symbols requested by market-data transforms.
type MarketSymbols = tuple[Symbol, ...]
#: Whether a schema column can contain null values.
type IsNullable = bool
#: Deterministic seed used by synthetic sampling.
type SyntheticSeed = int
#: Inclusive lower bound of a uniform synthetic distribution.
type DistributionMinimum = float
#: Inclusive upper bound of a uniform synthetic distribution.
type DistributionMaximum = float
#: Mean of a normal synthetic distribution.
type DistributionMean = float
#: Standard deviation of a normal synthetic distribution.
type DistributionStandardDeviation = float
#: Literal scalar supported by a constant synthetic distribution.
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

    #: Literal emitted for every generated row.
    value: SyntheticConstant


@record
class UniformDistribution:
    """Sample a numeric value uniformly over an inclusive range."""

    #: Inclusive lower bound of the sampled range.
    minimum: DistributionMinimum
    #: Inclusive upper bound of the sampled range.
    maximum: DistributionMaximum


@record
class NormalDistribution:
    """Sample a numeric value from a normal distribution."""

    #: Arithmetic mean of generated values.
    mean: DistributionMean
    #: Positive dispersion around the mean.
    standard_deviation: DistributionStandardDeviation


@record
class EntityValueDistribution:
    """Emit an entity value supplied by the configured transform."""

    #: Entity key whose configured value is emitted.
    key: EntityKey


@record
class PartitionTimestampDistribution:
    """Emit the timestamp of the partition being materialized."""


#: Sampling rule for one synthetic dataset column.
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

    #: Entity key represented by this set.
    key: EntityKey
    #: Concrete values emitted for the entity key.
    values: EntityValues


@record
class DatasetColumn:
    """One output column with a type and deterministic sampling distribution."""

    #: Stable output-column name.
    name: DatasetColumnName
    #: Spark-compatible type of the output column.
    column_type: DatasetColumnType
    #: Deterministic sampling rule for this output column.
    synthetic_distribution: SyntheticDistribution
    #: Whether generated values may be null.
    nullable: IsNullable = False


@record
class PositiveInvariant:
    """Require every value in one numeric column to be strictly positive."""

    #: Numeric column constrained to positive values.
    column: DatasetColumnName


@record
class LessThanOrEqualInvariant:
    """Require every lower-column value to be at most its upper-column value."""

    #: Column whose values must not exceed the upper column.
    lower_column: DatasetColumnName
    #: Column whose values bound the lower column.
    upper_column: DatasetColumnName


#: Cross-column constraint that synthetic samples must satisfy.
type SyntheticDistributionInvariant = (
    PositiveInvariant | LessThanOrEqualInvariant
)


@record
class DatasetSchema:
    """Static output contract for real and synthetic dataset transforms."""

    #: Complete physical output columns.
    columns: tuple[DatasetColumn, ...]
    #: Timestamp partition cadence for the dataset table.
    partition_type: PartitionType
    #: Cross-column constraints synthetic samples must satisfy.
    synthetic_distribution_invariants: tuple[
        SyntheticDistributionInvariant, ...
    ]
