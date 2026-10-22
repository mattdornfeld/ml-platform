"""Typed DataFrame validation shared by every materializer."""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Annotated, Mapping, Literal, assert_never

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
class PartitionValidationInput(ABC):
    """Static structural requirements common to one partitioned output frame."""

    frame: DataFrame
    partitioned_frame: DataFrame
    requested_partitions: DataFrame
    partition_column: str
    partition_interval: timedelta
    exact_columns: frozenset[str]
    non_null_columns: tuple[str, ...]
    unique_key_columns: tuple[str, ...]


@dataclass(frozen=True)
class DatasetValidationInput(PartitionValidationInput):
    """Validation contract for a dataset output frame."""


@dataclass(frozen=True)
class FeatureValidationInput(PartitionValidationInput):
    """Validation contract for a feature output frame and its input datasets."""

    feature_value_column: str
    feature_keys: tuple[str, ...]
    datasets: Mapping[str, Dataset]
    actual_schema: str
    expected_schema: str
    actual_partition_spec: str
    expected_partition_spec: str

    def __post_init__(self) -> None:
        """Freeze input dataset references used by feature-key validation."""
        object.__setattr__(self, "datasets", MappingProxyType(dict(self.datasets)))


@dataclass(frozen=True)
class CompletionValidationInput:
    """Committed output-partition receipts awaiting control-plane acknowledgement."""

    target_partition_keys: tuple[str, ...]
    content_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        """Freeze committed receipt data before validating it."""
        object.__setattr__(self, "content_hashes", MappingProxyType(dict(self.content_hashes)))


@dataclass(frozen=True)
class LabelValidationInput(PartitionValidationInput):
    """Validation contract for a label output frame."""

    actual_schema: str
    expected_schema: str
    actual_partition_spec: str
    expected_partition_spec: str


ValidationInput = CompletionValidationInput | DatasetValidationInput | FeatureValidationInput | LabelValidationInput


@dataclass(frozen=True)
class MaterializationValidator:
    """Validate a typed materialization result before its Iceberg write."""

    _validations: tuple[Validation, ...]

    def __init__(self, validations: list[Validation]) -> None:
        """Freeze the validation policy supplied by the materializer configuration."""
        object.__setattr__(self, "_validations", tuple(validations))

    def _validate_completion(self, validation_input: CompletionValidationInput) -> None:
        """Require committed rows for every partition selected by an all-partitions row rule."""
        requires_rows = any(
            isinstance(validation, MinimumRowsValidationConfig)
            and isinstance(validation.temporal_filter, AllPartitionsTemporalFilterConfig)
            for validation in self._validations
        )
        if not requires_rows:
            return
        missing = set(validation_input.target_partition_keys).difference(validation_input.content_hashes)
        if missing:
            raise ValueError(f"Transform produced no rows for required partitions: {sorted(missing)}.")

    def _configured(self, validation_type: type[ValidationConfig]) -> list[ValidationConfig]:
        """Return every declared validation with the requested concrete type."""
        return [validation for validation in self._validations if isinstance(validation, validation_type)]

    def _validate_partitioned_output(self, validation_input: PartitionValidationInput) -> None:
        """Apply structural and partition-level checks common to all resource outputs."""
        self._exact_columns(validation_input.frame, validation_input.exact_columns)
        self._non_null_columns(validation_input.frame, validation_input.non_null_columns)
        self._unique_key(validation_input.frame, validation_input.unique_key_columns)
        self._partition_coverage(validation_input)

    def _partition_coverage(self, validation_input: PartitionValidationInput) -> None:
        """Reject unexpected output and apply selected partition-level checks."""
        requested_keys = validation_input.requested_partitions.select(validation_input.partition_column).distinct()
        emitted_keys = validation_input.partitioned_frame.select(validation_input.partition_column).distinct()
        unexpected = emitted_keys.join(requested_keys, validation_input.partition_column, "left_anti")
        if unexpected.limit(1).count():
            raise ValueError("Output emitted rows outside requested partitions.")

        counts = validation_input.partitioned_frame.groupBy(validation_input.partition_column).count()
        for validation in self._configured(MinimumRowsValidationConfig):
            assert isinstance(validation, MinimumRowsValidationConfig)
            selected = self._filtered_partitions(
                requested_keys,
                validation.temporal_filter,
                validation_input.partition_column,
                validation_input.partition_interval,
            )
            selected_counts = selected.join(counts, validation_input.partition_column, "left").fillna({"count": 0})
            if selected_counts.where(F.col("count") < validation.minimum_rows).limit(1).count():
                raise ValueError(
                    f"Output emitted fewer than {validation.minimum_rows} rows for a selected partition."
                )

    def _validate_finite_feature_value(self, validation_input: FeatureValidationInput) -> None:
        """Reject non-finite feature values when the corresponding rule is declared."""
        if not self._configured(FiniteFeatureValueValidationConfig):
            return
        value = F.col(validation_input.feature_value_column).cast("double")
        invalid = validation_input.frame.where(F.isnan(value) | (value == float("inf")) | (value == float("-inf")))
        if invalid.limit(1).count():
            raise ValueError(f"Output emitted a non-finite {validation_input.feature_value_column}.")

    def _validate_feature_keys_from_inputs(self, validation_input: FeatureValidationInput) -> None:
        """Require feature keys to occur in a declared input dataset when configured."""
        if not self._configured(FeatureKeysFromInputValidationConfig):
            return
        input_key_frames: list[DataFrame] = []
        for dataset in validation_input.datasets.values():
            missing_keys = set(validation_input.feature_keys).difference(dataset.frame.columns)
            if missing_keys:
                raise ValueError(
                    f"Input dataset does not contain feature key columns: {', '.join(sorted(missing_keys))}."
                )
            input_key_frames.append(
                dataset.frame.select(
                    *(F.col(key).cast("string").alias(key) for key in validation_input.feature_keys)
                ).distinct()
            )
        if not input_key_frames:
            raise ValueError("Output requires at least one input dataset for feature-key validation.")
        input_keys = input_key_frames[0]
        for input_key_frame in input_key_frames[1:]:
            input_keys = input_keys.unionByName(input_key_frame)
        unknown = (
            validation_input.frame.select(*validation_input.feature_keys)
            .distinct()
            .join(input_keys.distinct(), list(validation_input.feature_keys), "left_anti")
        )
        if unknown.limit(1).count():
            raise ValueError("Output emitted feature-key values absent from input datasets.")

    @staticmethod
    def _exact_columns(frame: DataFrame, expected: frozenset[str]) -> None:
        """Reject output with missing or unexpected columns."""
        actual = set(frame.columns)
        if actual == expected:
            return
        details = []
        if missing := expected - actual:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unexpected := actual - expected:
            details.append(f"unexpected: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Output must return exactly {sorted(expected)} ({'; '.join(details)}).")

    @staticmethod
    def _non_null_columns(frame: DataFrame, columns: tuple[str, ...]) -> None:
        """Reject nulls left after values are cast to their persisted types."""
        null_predicate = F.col(columns[0]).isNull()
        for column in columns[1:]:
            null_predicate = null_predicate | F.col(column).isNull()
        if frame.where(null_predicate).limit(1).count():
            raise ValueError(f"Output must emit non-null, castable {', '.join(columns)} values.")

    @staticmethod
    def _unique_key(frame: DataFrame, columns: tuple[str, ...]) -> None:
        """Require at most one row for every logical row identity."""
        duplicates = frame.groupBy(*columns).count().where(F.col("count") > 1)
        if duplicates.limit(1).count():
            raise ValueError(f"Output emitted duplicate ({', '.join(columns)}) rows.")

    @staticmethod
    def _validate_layout(validation_input: FeatureValidationInput | LabelValidationInput) -> None:
        """Require a transform's static Iceberg declaration to match its job definition."""
        if validation_input.actual_schema != validation_input.expected_schema:
            raise ValueError("Transform iceberg_schema does not match its configured output definition.")
        if validation_input.actual_partition_spec != validation_input.expected_partition_spec:
            raise ValueError("Transform iceberg_partition_spec does not match its configured frequency.")

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

    def validate(self, validation_input: ValidationInput) -> None:
        """Apply mandatory and declared invariants for one resource output frame."""
        match validation_input:
            case CompletionValidationInput():
                self._validate_completion(validation_input)
            case DatasetValidationInput():
                self._validate_partitioned_output(validation_input)
            case FeatureValidationInput():
                self._validate_partitioned_output(validation_input)
                self._validate_layout(validation_input)
                self._validate_finite_feature_value(validation_input)
                self._validate_feature_keys_from_inputs(validation_input)
            case LabelValidationInput():
                self._validate_partitioned_output(validation_input)
                self._validate_layout(validation_input)
            case _ as unsupported_input:
                assert_never(unsupported_input)
