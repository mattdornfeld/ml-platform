"""Typed DataFrame validation shared by every materializer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Annotated, Literal, assert_never

import exchange_calendars as xcals
from exchange_calendars import ExchangeCalendar
import pandas as pd
from pydantic import Field
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk.materializers.exceptions import (
    DataFrameValidationError,
    ExactColumnsValidationError,
    FiniteValuesValidationError,
    InputFeatureKeysValidationError,
    MinimumRowsValidationError,
    NonNullValuesValidationError,
    RequestedPartitionsValidationError,
    TransformLayoutValidationError,
    UniqueKeyValidationError,
    UnsupportedValidationOverrideError,
)
from ml_platform.data_sdk.transforms._types import PartitionType


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
    reason: str = Field(min_length=1, description="Why this validation is required for the materialized data.")


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
    timestamps: tuple[datetime, ...] = Field(default=(), exclude=True,
                                             description="UTC output partition boundaries to check.")
    partition_type: PartitionType = Field(default=PartitionType.DAILY, exclude=True,
                                          description="Explicit cadence of the checked partitions.")
    timestamp_column: str = Field(default="timestamp", exclude=True, description="Output timestamp column name.")
    partition_column: str = Field(default="output_partition", exclude=True,
                                  description="Temporary logical partition column name.")


class NonNullValuesValidationConfig(ValidationConfig):
    """Require every declared persisted output column to be non-null."""

    type: Literal["non_null_values"] = "non_null_values"
    columns: tuple[str, ...] = Field(default=(), description="Output columns that must be non-null.")


class FiniteNumericValuesValidationConfig(ValidationConfig):
    """Require every declared persisted numeric output column to be finite."""

    type: Literal["finite_numeric_values"] = "finite_numeric_values"
    columns: tuple[str, ...] = Field(default=(), description="Output numeric columns that must be finite.")


class FeatureKeysFromInputValidationConfig(ValidationConfig):
    """Require feature keys to occur in at least one injected input dataset."""

    type: Literal["feature_keys_from_input"] = "feature_keys_from_input"
    feature_keys: tuple[str, ...] = Field(default=(), description="Output entity-key columns to validate.")
    dataset_table_identifiers: tuple[str, ...] = Field(default=(),
                                                       description="Iceberg input tables that define valid entity keys.")


class ExactColumnsValidationConfig(ValidationConfig):
    """Require an exact, ordered-independent set of output columns."""

    type: Literal["exact_columns"] = "exact_columns"
    columns: tuple[str, ...] = Field(default=(), description="Complete required output column set.")


class UniqueKeyValidationConfig(ValidationConfig):
    """Require each declared output key tuple to be unique."""

    type: Literal["unique_key"] = "unique_key"
    columns: tuple[str, ...] = Field(default=(), description="Columns that together identify a unique output row.")


class RequestedPartitionsValidationConfig(ValidationConfig):
    """Restrict output rows to declared timestamp partition boundaries."""

    type: Literal["requested_partitions"] = "requested_partitions"
    timestamps: tuple[datetime, ...] = Field(default=(), exclude=True,
                                             description="UTC output partition boundaries allowed in the data_frame.")
    partition_type: PartitionType = Field(default=PartitionType.DAILY, exclude=True,
                                          description="Explicit cadence of the checked partitions.")
    timestamp_column: str = Field(default="timestamp", exclude=True, description="Output timestamp column name.")
    partition_column: str = Field(default="output_partition", exclude=True,
                                  description="Temporary logical partition column name.")


DiscriminatedValidationConfig = Annotated[
    MinimumRowsValidationConfig
    | NonNullValuesValidationConfig
    | FiniteNumericValuesValidationConfig
    | FeatureKeysFromInputValidationConfig
    | ExactColumnsValidationConfig
    | UniqueKeyValidationConfig
    | RequestedPartitionsValidationConfig,
    Field(discriminator="type"),
]


@dataclass(frozen=True)
class DataFrameValidation(ABC):
    """One statically configured validation applied to a materialized DataFrame."""

    @abstractmethod
    def validate(self, data_frame: DataFrame) -> None:
        """Reject a DataFrame that violates this validation's explicit contract."""


@dataclass(frozen=True)
class ExactColumnsValidation(DataFrameValidation):
    """Require an output data_frame to contain exactly the declared columns."""

    expected_columns: frozenset[str]

    def validate(self, data_frame: DataFrame) -> None:
        """Reject missing or unexpected output columns."""
        actual_columns = set(data_frame.columns)
        if actual_columns == self.expected_columns:
            return
        details = []
        if missing_columns := self.expected_columns - actual_columns:
            details.append(f"missing: {', '.join(sorted(missing_columns))}")
        if unexpected_columns := actual_columns - self.expected_columns:
            details.append(f"unexpected: {', '.join(sorted(unexpected_columns))}")
        raise ExactColumnsValidationError(
            "Output columns differ from the declared schema "
            f"(expected: {', '.join(sorted(self.expected_columns))}; {'; '.join(details)})."
        )


@dataclass(frozen=True)
class NonNullColumnsValidation(DataFrameValidation):
    """Require persisted values in specified columns to be non-null."""

    columns: tuple[str, ...]

    def validate(self, data_frame: DataFrame) -> None:
        """Reject null values after materializer type coercion."""
        null_predicate = F.col(self.columns[0]).isNull()
        for column in self.columns[1:]:
            null_predicate = null_predicate | F.col(column).isNull()
        if data_frame.where(null_predicate).limit(1).count():
            raise NonNullValuesValidationError(
                f"Output contains null or uncastable values in required columns: {', '.join(self.columns)}."
            )


@dataclass(frozen=True)
class UniqueKeyValidation(DataFrameValidation):
    """Require each logical key tuple to identify at most one output row."""

    columns: tuple[str, ...]

    def validate(self, data_frame: DataFrame) -> None:
        """Reject duplicate logical output rows."""
        duplicates = data_frame.groupBy(*self.columns).count().where(F.col("count") > 1)
        if duplicates.limit(1).count():
            raise UniqueKeyValidationError(
                f"Output contains duplicate rows for logical key: {', '.join(self.columns)}."
            )


@dataclass(frozen=True)
class RequestedPartitionsValidation(DataFrameValidation):
    """Require output rows to belong to requested timestamp partitions."""

    requested_partitions: DataFrame
    timestamp_column: str
    partition_column: str
    partition_type: PartitionType

    def validate(self, data_frame: DataFrame) -> None:
        """Reject output emitted outside the requested partition set."""
        output_partitions = data_frame.select(
            F.date_trunc(self.partition_type.partition_unit, F.col(self.timestamp_column)).alias(self.partition_column)
        ).distinct()
        requested_partitions = self.requested_partitions.select(self.partition_column).distinct()
        unexpected_partitions = output_partitions.join(
            requested_partitions,
            self.partition_column,
            "left_anti",
        )
        if unexpected_partitions.limit(1).count():
            raise RequestedPartitionsValidationError(
                f"Output contains rows outside requested {self.partition_type.value} partitions "
                f"using timestamp column {self.timestamp_column!r}."
            )


@dataclass(frozen=True)
class MinimumRowsValidation(DataFrameValidation):
    """Require a configured number of rows in every selected requested partition."""

    requested_partitions: DataFrame
    timestamp_column: str
    partition_column: str
    partition_type: PartitionType
    minimum_rows: int
    temporal_filter: TemporalFilter

    def validate(self, data_frame: DataFrame) -> None:
        """Count rows in selected requested partitions and reject short output."""
        partitioned_frame = data_frame.withColumn(
            self.partition_column,
            F.date_trunc(self.partition_type.partition_unit, F.col(self.timestamp_column)),
        )
        counts = partitioned_frame.groupBy(self.partition_column).count()
        requested_partitions = self.requested_partitions.select(self.partition_column).distinct()
        selected_partitions = self._selected_partitions(requested_partitions)
        selected_counts = selected_partitions.join(counts, self.partition_column, "left").fillna({"count": 0})
        if selected_counts.where(F.col("count") < self.minimum_rows).limit(1).count():
            raise MinimumRowsValidationError(
                f"Output has fewer than {self.minimum_rows} rows in at least one selected "
                f"{self.partition_type.value} partition."
            )

    def _selected_partitions(self, requested_partitions: DataFrame) -> DataFrame:
        """Apply this validation's temporal filter to requested partition timestamps."""
        match self.temporal_filter:
            case AllPartitionsTemporalFilterConfig():
                return requested_partitions
            case XnysRegularSessionTemporalFilterConfig():
                calendar = xcals.get_calendar("XNYS")
                selected_timestamps = [
                    row[self.partition_column]
                    for row in requested_partitions.collect()
                    if self._overlaps_xnys_regular_session(
                        row[self.partition_column],
                        self.partition_type.partition_interval,
                        calendar,
                    )
                ]
                return requested_partitions.sparkSession.createDataFrame(
                    [(timestamp,) for timestamp in selected_timestamps],
                    requested_partitions.schema,
                )
            case _ as unsupported_filter:
                assert_never(unsupported_filter)

    @staticmethod
    def _overlaps_xnys_regular_session(
            partition_timestamp: datetime,
            partition_interval: timedelta,
            calendar: ExchangeCalendar,
    ) -> bool:
        """Return whether one UTC partition interval intersects XNYS regular hours."""
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


@dataclass(frozen=True)
class FiniteValuesValidation(DataFrameValidation):
    """Require values in every declared numeric output column to be finite."""

    columns: tuple[str, ...]

    def validate(self, data_frame: DataFrame) -> None:
        """Reject NaN and positive or negative infinity numeric values."""
        first_value = F.col(self.columns[0]).try_cast("double")
        invalid_predicate = (
            first_value.isNull()
            | F.isnan(first_value)
            | (first_value == float("inf"))
            | (first_value == float("-inf"))
        )
        for column in self.columns[1:]:
            value = F.col(column).try_cast("double")
            invalid_predicate = (
                invalid_predicate
                | value.isNull()
                | F.isnan(value)
                | (value == float("inf"))
                | (value == float("-inf"))
            )
        if data_frame.where(invalid_predicate).limit(1).count():
            raise FiniteValuesValidationError(
                f"Output contains NaN, infinity, or uncastable numeric values in: {', '.join(self.columns)}."
            )


@dataclass(frozen=True)
class InputFeatureKeysValidation(DataFrameValidation):
    """Require feature keys to occur in at least one injected dataset."""

    feature_keys: tuple[str, ...]
    input_frames: tuple[DataFrame, ...]

    def validate(self, data_frame: DataFrame) -> None:
        """Reject feature keys not represented by an injected input dataset."""
        input_key_frames: list[DataFrame] = []
        for input_frame in self.input_frames:
            missing_keys = set(self.feature_keys).difference(input_frame.columns)
            if missing_keys:
                raise InputFeatureKeysValidationError(
                    "Input dataset is missing required feature-key columns: "
                    f"{', '.join(sorted(missing_keys))}."
                )
            input_key_frames.append(
                input_frame.select(*(F.col(key).cast("string").alias(key) for key in self.feature_keys)).distinct()
            )
        if not input_key_frames:
            raise InputFeatureKeysValidationError(
                "Feature-key validation requires at least one injected input dataset."
            )
        input_keys = input_key_frames[0]
        for input_key_frame in input_key_frames[1:]:
            input_keys = input_keys.unionByName(input_key_frame)
        unknown_keys = data_frame.select(*self.feature_keys).distinct().join(
            input_keys.distinct(),
            list(self.feature_keys),
            "left_anti",
        )
        if unknown_keys.limit(1).count():
            raise InputFeatureKeysValidationError(
                f"Output contains feature-key values absent from input datasets for keys: {', '.join(self.feature_keys)}."
            )


@dataclass(frozen=True)
class TransformLayoutValidation(DataFrameValidation):
    """Require a transform's static Iceberg layout to match materializer configuration."""

    actual_schema: str
    expected_schema: str
    actual_partition_spec: str
    expected_partition_spec: str

    def validate(self, data_frame: DataFrame) -> None:
        """Reject an output transform whose declared storage layout differs from configuration."""
        del data_frame
        if self.actual_schema != self.expected_schema:
            raise TransformLayoutValidationError(
                "Transform Iceberg schema differs from the configured output definition "
                f"(expected: {self.expected_schema}; actual: {self.actual_schema})."
            )
        if self.actual_partition_spec != self.expected_partition_spec:
            raise TransformLayoutValidationError(
                "Transform Iceberg partition specification differs from the configured frequency "
                f"(expected: {self.expected_partition_spec}; actual: {self.actual_partition_spec})."
            )


@dataclass(frozen=True)
class ValidationConfigs:
    """Resolved default and override validation definitions for one materialization."""

    default_validation_configs: tuple[ValidationConfig, ...]
    override_validation_configs: tuple[ValidationConfig, ...]

    @staticmethod
    def _to_dict(validation_configs: tuple[ValidationConfig, ...]) -> dict[str, ValidationConfig]:
        return {config.type: config for config in validation_configs}

    @cached_property
    def resolved_validation_configs(self) -> tuple[ValidationConfig, ...]:
        default_configs_by_type = self._to_dict(self.default_validation_configs)
        overrides_by_type = self._to_dict(self.override_validation_configs)
        unsupported_types = set(overrides_by_type).difference(default_configs_by_type)
        if unsupported_types:
            raise UnsupportedValidationOverrideError(
                "Validation overrides are not part of the materializer default policy: "
                f"{', '.join(sorted(unsupported_types))}."
            )
        return tuple(
            default_config.model_copy(
                update=overrides_by_type[default_config.type].model_dump(exclude_unset=True)
            )
            if default_config.type in overrides_by_type
            else default_config
            for default_config in self.default_validation_configs
        )


@dataclass(frozen=True)
class MaterializationValidator:
    """Dispatch declarative validation configurations against one output DataFrame."""

    validation_configs: ValidationConfigs

    def _requested_partitions(self, data_frame: DataFrame, config: RequestedPartitionsValidationConfig) -> DataFrame:
        """Build the requested partition relation declared by one config."""
        return data_frame.sparkSession.createDataFrame(
            [(timestamp,) for timestamp in config.timestamps],
            f"{config.partition_column} timestamp",
        ).distinct()

    def validate(self, data_frame: DataFrame) -> None:
        """Execute every configured validation against the supplied output data_frame."""
        for config in self.validation_configs.resolved_validation_configs:
            try:
                match config:
                    case ExactColumnsValidationConfig():
                        ExactColumnsValidation(frozenset(config.columns)).validate(data_frame)
                    case NonNullValuesValidationConfig():
                        NonNullColumnsValidation(config.columns).validate(data_frame)
                    case FiniteNumericValuesValidationConfig():
                        FiniteValuesValidation(config.columns).validate(data_frame)
                    case UniqueKeyValidationConfig():
                        UniqueKeyValidation(config.columns).validate(data_frame)
                    case RequestedPartitionsValidationConfig():
                        requested = self._requested_partitions(data_frame, config)
                        RequestedPartitionsValidation(requested, config.timestamp_column, config.partition_column,
                                                      config.partition_type).validate(data_frame)
                    case MinimumRowsValidationConfig():
                        requested = self._requested_partitions(
                            data_frame,
                            RequestedPartitionsValidationConfig(
                                reason=config.reason,
                                timestamps=config.timestamps,
                                partition_type=config.partition_type,
                                timestamp_column=config.timestamp_column,
                                partition_column=config.partition_column,
                            ),
                        )
                        MinimumRowsValidation(
                            requested,
                            config.timestamp_column,
                            config.partition_column,
                            config.partition_type,
                            config.minimum_rows,
                            config.temporal_filter,
                        ).validate(data_frame)
                    case FeatureKeysFromInputValidationConfig():
                        input_frames = tuple(
                            data_frame.sparkSession.table(identifier)
                            for identifier in config.dataset_table_identifiers
                        )
                        InputFeatureKeysValidation(config.feature_keys, input_frames).validate(data_frame)
                    case _ as unsupported_config:
                        assert_never(unsupported_config)
            except DataFrameValidationError as error:
                raise type(error)(f"{config.reason}: {error}") from error
