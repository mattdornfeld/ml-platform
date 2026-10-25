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
#: Deterministic seed used by synthetic sampling.
SyntheticSeed = NewType("SyntheticSeed", int)
#: Inclusive lower bound of a uniform synthetic distribution.
DistributionMinimum = NewType("DistributionMinimum", float)
#: Inclusive upper bound of a uniform synthetic distribution.
DistributionMaximum = NewType("DistributionMaximum", float)
#: Mean of a normal synthetic distribution.
DistributionMean = NewType("DistributionMean", float)
#: Standard deviation of a normal synthetic distribution.
DistributionStandardDeviation = NewType("DistributionStandardDeviation", float)
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
    """Emit one deterministic constant for every generated row.

    Attributes:
        value: Literal emitted for every generated row.
    """

    value: SyntheticConstant


@record
class UniformDistribution:
    """Sample a numeric value uniformly over an inclusive range.

    Attributes:
        minimum: Inclusive lower bound of the sampled range.
        maximum: Inclusive upper bound of the sampled range.
    """

    minimum: DistributionMinimum
    maximum: DistributionMaximum


@record
class NormalDistribution:
    """Sample a numeric value from a normal distribution.

    Attributes:
        mean: Arithmetic mean of generated values.
        standard_deviation: Positive dispersion around the mean.
    """

    mean: DistributionMean
    standard_deviation: DistributionStandardDeviation


@record
class EntityValueDistribution:
    """Emit an entity value supplied by the configured transform.

    Attributes:
        key: Entity key whose configured value is emitted.
    """

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
    """Configured values emitted for one entity-key column.

    Attributes:
        key: Entity key represented by this set.
        values: Concrete values emitted for the entity key.
    """

    key: EntityKey
    values: EntityValues


@record
class DatasetColumn:
    """One output column with a type and deterministic sampling distribution.

    Attributes:
        name: Stable output-column name.
        column_type: Spark-compatible type of the output column.
        synthetic_distribution: Deterministic sampling rule for this column.
        nullable: Whether generated values may be null.
    """

    name: DatasetColumnName
    column_type: DatasetColumnType
    synthetic_distribution: SyntheticDistribution
    nullable: bool = False


@record
class PositiveInvariant:
    """Require every value in one numeric column to be strictly positive.

    Attributes:
        column: Numeric column constrained to positive values.
    """

    column: DatasetColumnName


@record
class LessThanOrEqualInvariant:
    """Require every lower-column value to be at most its upper-column value.

    Attributes:
        lower_column: Column whose values must not exceed the upper column.
        upper_column: Column whose values bound the lower column.
    """

    lower_column: DatasetColumnName
    upper_column: DatasetColumnName


#: Cross-column constraint that synthetic samples must satisfy.
type SyntheticDistributionInvariant = (
    PositiveInvariant | LessThanOrEqualInvariant
)


@record
class DatasetSchema:
    """Static output contract for real and synthetic dataset transforms.

    Attributes:
        columns: Complete physical output columns.
        partition_type: Timestamp partition cadence for the dataset table.
        synthetic_distribution_invariants: Cross-column sampling constraints.
    """

    columns: tuple[DatasetColumn, ...]
    partition_type: PartitionType
    synthetic_distribution_invariants: tuple[
        SyntheticDistributionInvariant, ...
    ]
