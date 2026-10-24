"""Schema-driven deterministic dataset transformation for tests."""

from typing import assert_never

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as spark_functions

from ml_platform.data_sdk._exceptions import DataSdkError
from ml_platform.data_sdk.transforms._dataset_schema import (
    ConstantDistribution,
    DatasetSchema,
    EntityValueDistribution,
    EntityValueSet,
    LessThanOrEqualInvariant,
    NormalDistribution,
    PartitionTimestampDistribution,
    PositiveInvariant,
    SchemaInvariant,
    SyntheticDistribution,
    SyntheticSeed,
    UniformDistribution,
)
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.utils import record

_MAXIMUM_SAMPLING_ATTEMPTS = 16


class SyntheticDataGenerationError(DataSdkError):
    """Raised when declared distributions cannot satisfy schema invariants."""


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
        """Sample schema columns until every row satisfies the invariants."""
        base_data_frame = self.spark.createDataFrame(
            [(timestamp,) for timestamp in self.partitions],
            "requested_timestamp timestamp",
        )
        for entity_values in self.entity_value_sets:
            base_data_frame = base_data_frame.crossJoin(
                self.spark.createDataFrame(
                    [(value,) for value in entity_values.values],
                    f"{entity_values.key.value} string",
                )
            )
        data_frame = self._sample(base_data_frame, self.synthetic_seed)
        for attempt in range(1, _MAXIMUM_SAMPLING_ATTEMPTS):
            candidate_data_frame = self._sample(
                base_data_frame,
                self.synthetic_seed + attempt,
            )
            data_frame = self._replace_invalid_rows(
                data_frame,
                candidate_data_frame,
            )
        if self._invalid_row_count(data_frame):
            raise SyntheticDataGenerationError(
                "Synthetic dataset distributions could not satisfy schema "
                f"invariants after {_MAXIMUM_SAMPLING_ATTEMPTS} attempts."
            )
        return data_frame.select(
            *(column.name for column in self.schema.columns)
        )

    def _sample(
        self, base_data_frame: DataFrame, seed: SyntheticSeed
    ) -> DataFrame:
        """Sample every declared output column for one deterministic attempt."""
        data_frame = base_data_frame
        for column_index, column in enumerate(self.schema.columns):
            data_frame = data_frame.withColumn(
                column.name,
                _sample_distribution(column.distribution, seed + column_index),
            )
        return data_frame

    def _replace_invalid_rows(
        self,
        data_frame: DataFrame,
        candidate_data_frame: DataFrame,
    ) -> DataFrame:
        """Keep valid rows and replace invalid rows with a fresh sample."""
        validity = self._validity_expression(data_frame)
        candidate_columns = [
            spark_functions.col(column.name).alias(f"candidate_{column.name}")
            for column in self.schema.columns
        ]
        candidate = candidate_data_frame.select(
            "requested_timestamp",
            *(
                entity_values.key.value
                for entity_values in self.entity_value_sets
            ),
            *candidate_columns,
        )
        key_columns = [
            "requested_timestamp",
            *(values.key.value for values in self.entity_value_sets),
        ]
        joined_data_frame = data_frame.join(candidate, on=key_columns)
        entity_column_names = {
            entity_values.key.value for entity_values in self.entity_value_sets
        }
        output_columns = (
            column
            for column in self.schema.columns
            if column.name not in entity_column_names
        )
        return joined_data_frame.select(
            *key_columns,
            *(
                spark_functions.when(validity, spark_functions.col(column.name))
                .otherwise(spark_functions.col(f"candidate_{column.name}"))
                .alias(column.name)
                for column in output_columns
            ),
        )

    def _invalid_row_count(self, data_frame: DataFrame) -> int:
        """Count generated rows that still violate a declared invariant."""
        return data_frame.where(~self._validity_expression(data_frame)).count()

    def _validity_expression(self, data_frame: DataFrame) -> Column:
        """Build the conjunction of every declared row-level invariant."""
        validity = spark_functions.expr("true")
        for invariant in self.schema.invariants:
            validity = validity & _invariant_expression(invariant, data_frame)
        return validity


def _sample_distribution(
    distribution: SyntheticDistribution,
    seed: SyntheticSeed,
) -> Column:
    """Build one deterministic Spark expression from a declared distribution."""
    match distribution:
        case ConstantDistribution(value=value):
            return spark_functions.lit(value)
        case UniformDistribution(minimum=minimum, maximum=maximum):
            return spark_functions.rand(seed) * (maximum - minimum) + minimum
        case NormalDistribution(
            mean=mean, standard_deviation=standard_deviation
        ):
            return spark_functions.randn(seed) * standard_deviation + mean
        case EntityValueDistribution(key=key):
            return spark_functions.col(key.value)
        case PartitionTimestampDistribution():
            return spark_functions.col("requested_timestamp")
        case _ as unsupported_distribution:
            assert_never(unsupported_distribution)


def _invariant_expression(
    invariant: SchemaInvariant,
    data_frame: DataFrame,
) -> Column:
    """Build a boolean Spark expression for one declared schema invariant."""
    match invariant:
        case PositiveInvariant(column=column):
            return data_frame[column] > spark_functions.lit(0)
        case LessThanOrEqualInvariant(
            lower_column=lower_column,
            upper_column=upper_column,
        ):
            return data_frame[lower_column] <= data_frame[upper_column]
        case _ as unsupported_invariant:
            assert_never(unsupported_invariant)
