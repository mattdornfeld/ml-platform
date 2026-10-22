"""Iceberg label materializer dispatched by the control plane."""

from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, cast, get_type_hints

from pydantic import Field, field_validator
from simple_classproperty import classproperty
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ml_platform.data_sdk.materializers._base import Materializer, MaterializerConfig, MaterializerPartition, WriteMode
from ml_platform.data_sdk.materializers._materialization_validator import MaterializationValidator
from ml_platform.data_sdk.transforms._label_transform import LabelTransform
from ml_platform.data_sdk.transforms._types import DataType, Dataset, DatasetDependency, FeatureKey, PartitionType


class LabelMaterializationConfig(MaterializerConfig):
    """Complete JSON contract consumed by a label Spark materializer."""

    dtype: str = Field(default="int", pattern=r"^(double|float|int|bigint)$", description="Spark SQL type used to persist label values.")
    frequency: str = Field(default="daily", pattern=r"^(hourly|daily)$", description="Cadence of the label's logical output partitions.")
    feature_keys: list[str] = Field(default_factory=lambda: ["symbol"], min_length=1, description="Ordered entity-key columns that identify each label value.")
    datasets: list[DatasetDependency] = Field(min_length=1, description="Versioned dataset dependencies consumed by the label transform.")
    transform: str = Field(
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
        description="Absolute import path of the LabelTransform implementation.",
    )
    parameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict, description="Typed keyword arguments passed to the label transform constructor."
    )

    @field_validator("partitions")
    @classmethod
    def _require_partition_dependencies(cls, partitions: list[MaterializerPartition]) -> list[MaterializerPartition]:
        if any(not partition.partition_dependencies for partition in partitions):
            raise ValueError("Label partitions must include pinned partition_dependencies.")
        return partitions

    @field_validator("feature_keys")
    @classmethod
    def _unique_feature_keys(cls, feature_keys: list[str]) -> list[str]:
        """Require an unambiguous label row identity."""
        if len(set(feature_keys)) != len(feature_keys):
            raise ValueError("Feature keys must be unique.")
        return feature_keys


@dataclass(frozen=True)
class LabelResource:
    """Identity for one materialized label resource."""

    name: str
    version: int


@dataclass(frozen=True)
class LabelMaterializer(Materializer[LabelMaterializationConfig]):
    """Construct, validate, persist, and report one label materialization."""

    @classproperty
    def config_model(cls) -> type[LabelMaterializationConfig]:
        """Return the strict runtime contract for label jobs."""
        return LabelMaterializationConfig

    @classproperty
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for label jobs."""
        return "ml-platform-label-materializer"

    @classproperty
    def iceberg_namespace(cls) -> str:
        """Store label resources in their dedicated Iceberg namespace."""
        return "labels"

    @classproperty
    def callback_resource(cls) -> str:
        """Report label partition receipts as label resources."""
        return "label"

    @classproperty
    def write_mode(cls) -> WriteMode:
        """Merge labels by their timestamp and configured feature-key identity."""
        return WriteMode.MERGE

    datasets: Mapping[str, Dataset]

    def __post_init__(self) -> None:
        """Freeze a snapshot of the datasets supplied at construction."""
        object.__setattr__(self, "datasets", MappingProxyType(dict(self.datasets)))

    def _partition_receipt_data(self, partition: MaterializerPartition) -> dict[str, object]:
        """Pin the input dataset revisions used for this label partition."""
        return {"dataset_partition_ids": [partition_dependency.partition_id for partition_dependency in partition.partition_dependencies]}

    @property
    def _feature_key_columns(self) -> tuple[str, ...]:
        """Return the immutable configured entity columns for one label row."""
        return tuple(self.config.feature_keys)

    def _prepare_frame(self, frame: DataFrame) -> DataFrame:
        """Cast a strategy result to the canonical label table schema."""
        expected_columns = {"timestamp", *self._feature_key_columns, "label_value"}
        self._validator.exact_columns(frame, expected_columns)
        return frame.select(
            F.col("timestamp").try_cast("timestamp").alias("timestamp"),
            *(F.col(key).cast("string").alias(key) for key in self._feature_key_columns),
            F.col("label_value").try_cast(self.config.dtype).alias("label_value"),
        )

    def _validate_declared_layout(self, transform: LabelTransform) -> None:
        """Require checked-in label metadata to agree with the transform's static layout."""
        expected_schema = (
            "timestamp TIMESTAMP, "
            f"{', '.join(f'{key} STRING' for key in self._feature_key_columns)}, "
            f"label_value {self.config.dtype.upper()}"
        )
        expected_partition_spec = {"hourly": "hours(timestamp)", "daily": "days(timestamp)"}[self.config.frequency]
        self._validator.declared_layout(
            actual_schema=transform.iceberg_schema,
            expected_schema=expected_schema,
            actual_partition_spec=transform.iceberg_partition_spec,
            expected_partition_spec=expected_partition_spec,
        )

    def _validate_frame(self, frame: DataFrame, spark: SparkSession, partition_spec: str) -> None:
        """Validate the canonical label contract before its Iceberg write."""
        required_columns = ("timestamp", *self._feature_key_columns, "label_value")
        self._validator.non_null_columns(frame, required_columns)
        self._validator.unique_key(frame, ("timestamp", *self._feature_key_columns))
        partitioned = frame.withColumn("output_partition", F.date_trunc(self._partition_unit(partition_spec), F.col("timestamp")))
        requested = spark.createDataFrame(
            [(self._partition_timestamp(partition.timestamp, partition_spec),) for partition in self.config.partitions],
            "output_partition timestamp",
        ).distinct()
        self._validator.partition_coverage(partitioned, requested, partition_column="output_partition")

    def _row_digest(self) -> F.Column:
        """Hash the canonical persisted fields for one label row."""
        columns = ("timestamp", *self._feature_key_columns, "label_value")
        return F.sha2(F.to_json(F.struct(*(F.col(column) for column in columns))), 256)

    def _write_frame(self, frame: DataFrame) -> None:
        """Merge labels using their canonical timestamp and feature-key identity."""
        self._merge_frame(frame, self._table_identifier, ("timestamp", *self._feature_key_columns))

    @classmethod
    def _label_id(cls, resource: LabelResource) -> str:
        """Return the physical versioned label table name."""
        return cls.config_model._iceberg_table_name_for(resource.name, resource.version)

    @classmethod
    def _table_identifier_for(cls, resource: LabelResource) -> str:
        """Return the Iceberg identifier for one label resource."""
        return cls._iceberg_table_identifier(resource.name, resource.version)

    @classmethod
    def _storage_path(cls, warehouse_dir: str | Path, resource: LabelResource) -> Path:
        """Return the resource's versioned warehouse directory."""
        return Path(warehouse_dir).resolve() / cls.iceberg_namespace / cls._label_id(resource)

    @staticmethod
    def _load_transform(import_path: str) -> type[LabelTransform]:
        """Load a LabelTransform class referenced by the API-delivered definition."""
        module_name, class_name = import_path.rsplit(".", maxsplit=1)
        try:
            transform = getattr(import_module(module_name), class_name)
        except (AttributeError, ImportError) as error:
            raise ValueError(f"Cannot load label transform: {import_path}") from error
        if not isinstance(transform, type) or not issubclass(transform, LabelTransform):
            raise ValueError(f"Label transform must inherit LabelTransform: {import_path}")
        return cast(type[LabelTransform], transform)

    def _instantiate_transform(
        self, transform_class: type[LabelTransform], parameters: dict[str, object], datasets: Mapping[str, Dataset]
    ) -> LabelTransform:
        """Require the transform constructor to match declared datasets and parameters."""
        constructor_parameters = {
            name: parameter
            for name, parameter in signature(transform_class).parameters.items()
            if parameter.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        }
        annotations = get_type_hints(transform_class.__init__)
        injected_parameters = {
            "keys": [FeatureKey(key) for key in self.config.feature_keys],
            "partition_type": PartitionType(self.config.frequency),
            "data_type": DataType(self.config.dtype),
        }
        dataset_parameters = {name for name in constructor_parameters if annotations.get(name) is Dataset}
        if dataset_parameters != set(datasets):
            raise ValueError(
                "Label transform dataset discrepancy "
                f"(constructor: {', '.join(sorted(dataset_parameters)) or 'none'}; "
                f"declared: {', '.join(sorted(datasets)) or 'none'})."
            )
        parameter_definitions = {
            name: parameter
            for name, parameter in constructor_parameters.items()
            if name not in dataset_parameters and name not in injected_parameters
        }
        required = {name for name, parameter in parameter_definitions.items() if parameter.default is Parameter.empty}
        missing, unexpected = required.difference(parameters), set(parameters).difference(parameter_definitions)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if unexpected:
                details.append(f"unexpected: {', '.join(sorted(unexpected))}")
            raise ValueError(f"Label transform parameter discrepancy ({'; '.join(details)}).")
        for name, value in parameters.items():
            expected_type = annotations.get(name)
            if isinstance(expected_type, type) and type(value) is not expected_type:
                raise ValueError(
                    "Label transform parameter discrepancy "
                    f"(type for {name}: expected {expected_type.__name__}, got {type(value).__name__})."
                )
        try:
            return transform_class(**datasets, **injected_parameters, **parameters)
        except TypeError as error:
            raise ValueError(f"Label transform parameter discrepancy for {transform_class.__name__}: {error}") from error

    def _materialize_transform(self, transform: LabelTransform, *, spark: SparkSession) -> int:
        """Persist one transform result through the label table lifecycle."""
        schema, partition_spec = self._iceberg_layout(transform)
        labels = self._prepare_frame(transform().localCheckpoint(eager=True))
        self._validate_frame(labels, spark, partition_spec)
        self._ensure_table(spark, schema=schema, partition_spec=partition_spec)
        self._write_frame(labels)
        return labels.count()

    def _create_transform(self, spark: SparkSession) -> LabelTransform:
        """Instantiate the configured label transform with its pinned inputs."""
        transform = self._instantiate_transform(
            self._load_transform(self.config.transform), self.config.parameters, self.datasets
        )
        self._validate_declared_layout(transform)
        return transform

    def materialize(self, spark: SparkSession, transform: LabelTransform) -> DataFrame:
        """Return the configured transform's checkpointed label DataFrame."""
        return transform().localCheckpoint(eager=True)
