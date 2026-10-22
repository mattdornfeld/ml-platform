"""Typed DataFrame validation shared by every materializer."""

from dataclasses import dataclass

from pydantic import Field
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk.transforms._types import Dataset


class MaterializerValidations(BaseConfig):
    """Explicit validation policy applied to every requested output partition."""

    min_rows_per_partition: int = Field(
        default=1,
        ge=1,
        description="Minimum rows required for each non-empty requested output partition.",
    )
    allow_empty_partitions: bool = Field(
        default=False,
        description="Whether requested output partitions may have no emitted rows and receive empty receipts.",
    )
    require_finite_feature_value: bool = Field(
        default=True,
        description="Whether feature values must be finite rather than NaN or infinite.",
    )
    require_feature_keys_from_input: bool = Field(
        default=True,
        description="Whether output feature keys must be present in the declared input datasets.",
    )


@dataclass(frozen=True)
class MaterializationValidator:
    """Validate prepared DataFrames against one materialization job's policy.

    The validator never mutates or persists a DataFrame. It raises ``ValueError``
    before a write when a required schema, key, value, or partition invariant fails.
    Empty requested partitions are permitted only when explicitly configured.
    """

    validations: MaterializerValidations
    context: str

    def required_columns(self, frame: DataFrame, required: set[str]) -> None:
        """Reject output that omits any required column."""
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{self.context} omitted required columns: {', '.join(sorted(missing))}.")

    def exact_columns(self, frame: DataFrame, expected: set[str]) -> None:
        """Reject output with missing or unexpected columns."""
        actual = set(frame.columns)
        if actual == expected:
            return
        details = []
        if missing := expected - actual:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unexpected := actual - expected:
            details.append(f"unexpected: {', '.join(sorted(unexpected))}")
        raise ValueError(f"{self.context} must return exactly {sorted(expected)} ({'; '.join(details)}).")

    def non_null_columns(self, frame: DataFrame, columns: tuple[str, ...]) -> None:
        """Reject nulls left after values are cast to their persisted types."""
        null_predicate = F.col(columns[0]).isNull()
        for column in columns[1:]:
            null_predicate = null_predicate | F.col(column).isNull()
        if frame.where(null_predicate).limit(1).count():
            raise ValueError(f"{self.context} must emit non-null, castable {', '.join(columns)} values.")

    def unique_key(self, frame: DataFrame, columns: tuple[str, ...]) -> None:
        """Require at most one row for every logical row identity."""
        duplicates = frame.groupBy(*columns).count().where(F.col("count") > 1)
        if duplicates.limit(1).count():
            raise ValueError(f"{self.context} emitted duplicate ({', '.join(columns)}) rows.")

    def partition_coverage(self, frame: DataFrame, requested: DataFrame, *, partition_column: str) -> None:
        """Reject unexpected output and enforce requested-partition row policy."""
        requested_keys = requested.select(partition_column).distinct()
        emitted_keys = frame.select(partition_column).distinct()
        missing = requested_keys.join(emitted_keys, partition_column, "left_anti")
        unexpected = emitted_keys.join(requested_keys, partition_column, "left_anti")
        if unexpected.limit(1).count():
            raise ValueError(f"{self.context} output partitions do not exactly match requested partitions.")
        if missing.limit(1).count() and not self.validations.allow_empty_partitions:
            raise ValueError(f"{self.context} output partitions do not exactly match requested partitions.")
        counts = frame.groupBy(partition_column).count()
        too_small = counts.where(F.col("count") < self.validations.min_rows_per_partition)
        if too_small.limit(1).count():
            raise ValueError(
                f"{self.context} emitted fewer than {self.validations.min_rows_per_partition} rows for a requested partition."
            )

    def non_empty(self, frame: DataFrame) -> None:
        """Require output rows unless the job explicitly permits empty partitions."""
        if not self.validations.allow_empty_partitions and not frame.limit(1).count():
            raise ValueError(f"{self.context} returned no rows.")

    def finite_column(self, frame: DataFrame, column: str) -> None:
        """Reject NaN and infinite values when finite values are required."""
        if not self.validations.require_finite_feature_value:
            return
        value = F.col(column).cast("double")
        invalid = frame.where(F.isnan(value) | (value == float("inf")) | (value == float("-inf")))
        if invalid.limit(1).count():
            raise ValueError(f"{self.context} emitted a non-finite {column}.")

    def feature_keys_from_inputs(
        self, frame: DataFrame, *, feature_keys: tuple[str, ...], datasets: dict[str, Dataset]
    ) -> None:
        """Require output entity keys to exist in one of the injected datasets."""
        if not self.validations.require_feature_keys_from_input:
            return
        input_key_frames: list[DataFrame] = []
        for dataset in datasets.values():
            missing_keys = set(feature_keys).difference(dataset.frame.columns)
            if missing_keys:
                raise ValueError(
                    f"Input dataset does not contain feature key columns: {', '.join(sorted(missing_keys))}."
                )
            input_key_frames.append(
                dataset.frame.select(*(F.col(key).cast("string").alias(key) for key in feature_keys)).distinct()
            )
        if not input_key_frames:
            raise ValueError(f"{self.context} requires at least one input dataset for feature-key validation.")
        input_keys = input_key_frames[0]
        for input_key_frame in input_key_frames[1:]:
            input_keys = input_keys.unionByName(input_key_frame)
        unknown = frame.select(*feature_keys).distinct().join(input_keys.distinct(), list(feature_keys), "left_anti")
        if unknown.limit(1).count():
            raise ValueError(f"{self.context} emitted feature-key values absent from input datasets.")

    def declared_layout(
        self,
        *,
        actual_schema: str,
        expected_schema: str,
        actual_partition_spec: str,
        expected_partition_spec: str,
    ) -> None:
        """Require a transform's static Iceberg declaration to match its job definition."""
        if actual_schema != expected_schema:
            raise ValueError(f"{self.context} iceberg_schema does not match its configured output definition.")
        if actual_partition_spec != expected_partition_spec:
            raise ValueError(f"{self.context} iceberg_partition_spec does not match its configured frequency.")
