"""Schema-driven deterministic dataset transform for tests."""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as spark_functions

from ml_platform.data_sdk.transforms._dataset_schema import (
    DatasetColumn,
    DatasetSchema,
    EntityValueSet,
    SyntheticSeed,
    SyntheticValueStrategy,
)
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.utils import record


@record
class SyntheticDatasetTransform(DatasetTransform):
    """Generate deterministic rows from a real transform's declared schema."""

    source_transform: DatasetTransform
    synthetic_seed: SyntheticSeed

    @property
    def schema(self) -> DatasetSchema:
        """Return the exact schema declared by the wrapped real transform."""
        return self.source_transform.schema

    @property
    def entity_value_sets(self) -> tuple[EntityValueSet, ...]:
        """Return entity values declared by the wrapped real transform."""
        return self.source_transform.entity_value_sets

    @property
    def value_schema(self) -> dict[str, str]:
        """Return the wrapped transform's typed non-key columns."""
        return self.source_transform.value_schema

    def __call__(self) -> DataFrame:
        """Generate rows for every requested partition and entity value."""
        data_frame = self.spark.createDataFrame(
            [(timestamp,) for timestamp in self.partitions],
            "requested_timestamp timestamp",
        )
        for entity_values in self.entity_value_sets:
            data_frame = data_frame.crossJoin(
                self.spark.createDataFrame(
                    [(value,) for value in entity_values.values],
                    f"{entity_values.key.value} string",
                )
            )
        for column in self.schema.columns:
            data_frame = data_frame.withColumn(
                column.name,
                _column_value(column, self.synthetic_seed),
            )
        return data_frame.select(
            *(column.name for column in self.schema.columns)
        )


def _column_value(
    column: DatasetColumn, synthetic_seed: SyntheticSeed
) -> Column:
    """Build the deterministic Spark expression for one declared schema column."""
    match column.synthetic_value_strategy:
        case SyntheticValueStrategy.PARTITION_TIMESTAMP:
            return spark_functions.col("requested_timestamp")
        case SyntheticValueStrategy.ENTITY_VALUE:
            return spark_functions.col(column.name)
        case SyntheticValueStrategy.MARKET_SESSION:
            return spark_functions.lit("regular")
        case SyntheticValueStrategy.POSITIVE_FLOAT:
            return spark_functions.lit(100.0 + synthetic_seed)
        case SyntheticValueStrategy.NONNEGATIVE_INTEGER:
            return spark_functions.lit(1_000 + synthetic_seed)
