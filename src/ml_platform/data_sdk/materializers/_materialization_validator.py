"""Typed DataFrame validation shared by every materializer."""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

import exchange_calendars as xcals
from exchange_calendars import ExchangeCalendar
import pandas as pd

from pydantic import Field
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk.transforms._types import Dataset


class TemporalFilterConfig(BaseConfig, ABC):
    """Select requested partitions to which a partition-level validation applies."""

    type: str = Field(description="Discriminator for the temporal filter implementation.")


class AllPartitionsTemporalFilterConfig(TemporalFilterConfig):
    """Apply a partition-level validation to every requested partition."""

    type: Literal["all_partitions"] = Field(
        default="all_partitions",
        description="Apply the validation to every requested partition.",
    )


class XnysRegularSessionTemporalFilterConfig(TemporalFilterConfig):
    """Select partitions that overlap an actual XNYS regular trading session."""

    type: Literal["xnys_regular_session"] = Field(
        default="xnys_regular_session",
        description="Select partitions overlapping the XNYS regular trading session.",
    )


TemporalFilter = Annotated[
    AllPartitionsTemporalFilterConfig | XnysRegularSessionTemporalFilterConfig,
    Field(discriminator="type"),
]


class ValidationConfig(BaseConfig, ABC):
    """Base type for one declared materialization validation."""

    type: str = Field(description="Discriminator for the validation implementation.")


class PartitionValidationConfig(ValidationConfig, ABC):
    """Validation applied to a selected subset of requested output partitions."""

    temporal_filter: TemporalFilter = Field(
        default_factory=AllPartitionsTemporalFilterConfig,
        description="Requested output partitions to which this validation applies.",
    )


class MinimumRowsValidationConfig(PartitionValidationConfig):
    """Require a minimum number of output rows in each selected partition."""

    type: Literal["minimum_rows"] = Field(default="minimum_rows", description="Require a minimum row count.")
    minimum_rows: int = Field(ge=1, description="Minimum output rows required in each selected partition.")


class FiniteFeatureValueValidationConfig(ValidationConfig):
    """Require feature values to be finite after casting to their configured dtype."""

    type: Literal["finite_feature_value"] = Field(
        default="finite_feature_value",
        description="Reject NaN and positive or negative infinity feature values.",
    )


class FeatureKeysFromInputValidationConfig(ValidationConfig):
    """Require feature keys to occur in at least one injected input dataset."""

    type: Literal["feature_keys_from_input"] = Field(
        default="feature_keys_from_input",
        description="Require feature-key values to occur in an injected input dataset.",
    )


Validation = Annotated[
    MinimumRowsValidationConfig | FiniteFeatureValueValidationConfig | FeatureKeysFromInputValidationConfig,
    Field(discriminator="type"),
]


@dataclass(frozen=True)
class MaterializationValidator:
    """Apply declared and mandatory DataFrame invariants before an Iceberg write.

    Required columns, exact layouts, non-null persisted values, unique logical keys,
    unexpected output partitions, and declared transform layouts are always enforced.
    Each configured ``ValidationConfig`` adds its own check; partition-level checks
    operate only on requested partitions selected by their ``temporal_filter``.
    """

    validations: list[Validation]
    context: str

    def _configured(self, validation_type: type[ValidationConfig]) -> list[ValidationConfig]:
        """Return every declared validation with the requested concrete type."""
        return [validation for validation in self.validations if isinstance(validation, validation_type)]

    @staticmethod
    def _filtered_partitions(
        requested: DataFrame,
        filter_config: TemporalFilter,
        partition_column: str,
        partition_interval: timedelta,
    ) -> DataFrame:
        """Return requested partitions selected by a statically declared temporal filter."""
        match filter_config:
            case AllPartitionsTemporalFilterConfig():
                return requested.select(partition_column).distinct()
            case XnysRegularSessionTemporalFilterConfig():
                calendar = xcals.get_calendar("XNYS")
                selected_timestamps = [
                    row[partition_column]
                    for row in requested.select(partition_column).distinct().collect()
                    if MaterializationValidator._overlaps_xnys_regular_session(
                        row[partition_column], partition_interval, calendar
                    )
                ]
                return requested.sparkSession.createDataFrame(
                    [(timestamp,) for timestamp in selected_timestamps],
                    requested.schema,
                )

    @staticmethod
    def _overlaps_xnys_regular_session(
        partition_timestamp: datetime,
        partition_interval: timedelta,
        calendar: ExchangeCalendar,
    ) -> bool:
        """Return whether one UTC output-partition interval intersects an XNYS session."""
        partition_start = (
            partition_timestamp.replace(tzinfo=timezone.utc)
            if partition_timestamp.tzinfo is None
            else partition_timestamp.astimezone(timezone.utc)
        )
        session = pd.Timestamp(partition_start.date())
        if not calendar.is_session(session):
            return False
        session_open = calendar.session_open(session).to_pydatetime()
        session_close = calendar.session_close(session).to_pydatetime()
        partition_end = partition_start + partition_interval
        return partition_start < session_close and partition_end > session_open

    def requires_rows_for_all_partitions(self) -> bool:
        """Return whether every requested partition must contain rows after writing."""
        return any(
            isinstance(validation, MinimumRowsValidationConfig)
            and isinstance(validation.temporal_filter, AllPartitionsTemporalFilterConfig)
            for validation in self.validations
        )

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

    def partition_coverage(
        self,
        frame: DataFrame,
        requested: DataFrame,
        *,
        partition_column: str,
        partition_interval: timedelta,
    ) -> None:
        """Reject unexpected output and apply each configured partition-level check."""
        requested_keys = requested.select(partition_column).distinct()
        emitted_keys = frame.select(partition_column).distinct()
        unexpected = emitted_keys.join(requested_keys, partition_column, "left_anti")
        if unexpected.limit(1).count():
            raise ValueError(f"{self.context} emitted output outside requested partitions.")

        counts = frame.groupBy(partition_column).count()
        for validation in self._configured(MinimumRowsValidationConfig):
            assert isinstance(validation, MinimumRowsValidationConfig)
            selected = self._filtered_partitions(
                requested_keys,
                validation.temporal_filter,
                partition_column,
                partition_interval,
            )
            selected_counts = selected.join(counts, partition_column, "left").fillna({"count": 0})
            if selected_counts.where(F.col("count") < validation.minimum_rows).limit(1).count():
                raise ValueError(
                    f"{self.context} emitted fewer than {validation.minimum_rows} rows for a selected partition."
                )

    def finite_column(self, frame: DataFrame, column: str) -> None:
        """Reject non-finite values when the finite-feature-value validation is configured."""
        if not self._configured(FiniteFeatureValueValidationConfig):
            return
        value = F.col(column).cast("double")
        invalid = frame.where(F.isnan(value) | (value == float("inf")) | (value == float("-inf")))
        if invalid.limit(1).count():
            raise ValueError(f"{self.context} emitted a non-finite {column}.")

    def feature_keys_from_inputs(
        self, frame: DataFrame, *, feature_keys: tuple[str, ...], datasets: dict[str, Dataset]
    ) -> None:
        """Require input membership when the corresponding validation is configured."""
        if not self._configured(FeatureKeysFromInputValidationConfig):
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
