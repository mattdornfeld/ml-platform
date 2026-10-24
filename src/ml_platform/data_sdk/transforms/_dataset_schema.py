"""Strongly typed schema declarations for dataset transforms."""

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


class DatasetColumnType(StrEnum):
    """Spark-compatible column types supported by synthetic data generation."""

    TIMESTAMP = "timestamp"
    STRING = "string"
    DOUBLE = "double"
    BIGINT = "bigint"


class SyntheticValueStrategy(StrEnum):
    """Declared deterministic generator for one synthetic dataset column."""

    PARTITION_TIMESTAMP = "partition_timestamp"
    ENTITY_VALUE = "entity_value"
    MARKET_SESSION = "market_session"
    POSITIVE_FLOAT = "positive_float"
    NONNEGATIVE_INTEGER = "nonnegative_integer"


@record
class EntityValueSet:
    """Configured values emitted for one entity-key column."""

    key: EntityKey
    values: EntityValues


@record
class DatasetColumn:
    """One declared dataset column and its deterministic synthetic generator."""

    name: DatasetColumnName
    column_type: DatasetColumnType
    nullable: IsNullable
    synthetic_value_strategy: SyntheticValueStrategy


@record
class DatasetSchema:
    """Static dataset output contract shared by real and synthetic transforms."""

    columns: tuple[DatasetColumn, ...]
    partition_type: PartitionType
