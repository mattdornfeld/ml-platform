"""Shared materialization base for resources backed by a configured data transform."""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from typing import TypeGuard, get_type_hints

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as spark_functions
from pyspark.sql.column import Column

from ml_platform.data_sdk.configs import TransformConfig
from ml_platform.data_sdk.materializers._base import (
    Materializer,
    MaterializerConfig,
    MaterializerPartition,
)
from ml_platform.data_sdk.materializers._iceberg_table import IcebergTable
from ml_platform.data_sdk.materializers._materialization_validator import (
    AllPartitionsTemporalFilterConfig,
    ExactColumnsValidationConfig,
    FiniteNumericValuesValidationConfig,
    MinimumRowsValidationConfig,
    NonNullValuesValidationConfig,
    UniqueKeyValidationConfig,
    ValidationConfig,
)
from ml_platform.data_sdk.materializers.exceptions import (
    DatasetPartitionPlanError,
    DuplicateDatasetArgumentError,
    TransformDatasetInjectionError,
    TransformImportError,
    TransformParameterError,
    TransformTypeError,
)
from ml_platform.data_sdk.transforms._base import DataTransform
from ml_platform.data_sdk.transforms._types import (
    Dataset,
    DatasetReference,
    DataType,
    FeatureDatasetDependency,
    FeatureKey,
    PartitionType,
)


class TransformMaterializationConfig(MaterializerConfig, TransformConfig):
    """Runtime transformation definition enriched with materialization-specific inputs."""


@dataclass(frozen=True)
class TransformTableDefinition:
    """Static table-specific values declared by each transform resource kind."""

    namespace: str
    resource_type: str
    value_column: str


class TransformMaterializer[
    Config: TransformMaterializationConfig,
    Transform: DataTransform,
](Materializer[Config], ABC):
    """Load API-pinned datasets and construct the declared concrete transform."""

    @property
    @abstractmethod
    def _transform_type(self) -> type[Transform]:
        """Return the specific transform contract accepted by this materializer."""

    @property
    @abstractmethod
    def _table_definition(self) -> TransformTableDefinition:
        """Return static table values declared by this transform resource type."""

    @property
    def _feature_key_columns(self) -> tuple[str, ...]:
        """Return the immutable configured entity columns for one transform row."""
        return tuple(
            feature_key.value for feature_key in self.config.feature_keys
        )

    @property
    def iceberg_table(self) -> IcebergTable:
        """Return the common Iceberg definition for this transform resource."""
        return IcebergTable(
            catalog=self.config.iceberg_catalog(),
            namespace=self._table_definition.namespace,
            name=self.config.name,
            version=self.config.version,
            resource_type=self._table_definition.resource_type,
            schema={
                "timestamp": "timestamp",
                **dict.fromkeys(self._feature_key_columns, "string"),
                self._table_definition.value_column: self.config.dtype,
            },
            partition_type=self.config.partition_type,
            description=self.config.description,
            dependencies=tuple(
                dataset.model_dump(mode="json")
                for dataset in self.config.datasets
            ),
            validations=tuple(
                validation.model_dump(mode="json")
                for validation in self.config.validations
            ),
            additional_properties=(
                ("transform.dtype", self.config.dtype),
                (
                    "transform.feature_keys",
                    json.dumps(self.config.feature_keys),
                ),
                ("transform.implementation", self.config.transform),
            ),
        )

    def _row_digest(self) -> Column:
        """Hash the canonical persisted columns for partition completion receipts."""
        columns = (
            "timestamp",
            *self._feature_key_columns,
            self._table_definition.value_column,
        )
        return spark_functions.sha2(
            spark_functions.to_json(
                spark_functions.struct(
                    *(spark_functions.col(column) for column in columns)
                )
            ),
            256,
        )

    @property
    def validation_configs(self) -> tuple[ValidationConfig, ...]:
        """Declare validations shared by every feature and label materialization."""
        value_column = self._table_definition.value_column
        return (
            MinimumRowsValidationConfig(
                reason="Transform partitions must contain at least one row.",
                minimum_rows=1,
                timestamps=self.config.partition_timestamps,
                partition_type=self.config.partition_type,
                temporal_filter=AllPartitionsTemporalFilterConfig(),
            ),
            ExactColumnsValidationConfig(
                reason="Transform output must use the canonical transform schema.",
                columns=("timestamp", *self._feature_key_columns, value_column),
            ),
            NonNullValuesValidationConfig(
                reason="Transform keys and values must be present and castable.",
                columns=("timestamp", *self._feature_key_columns, value_column),
            ),
            FiniteNumericValuesValidationConfig(
                reason="Transform values must be finite.",
                columns=(value_column,),
            ),
            UniqueKeyValidationConfig(
                reason="Each transform key may have one value per timestamp.",
                columns=("timestamp", *self._feature_key_columns),
            ),
        )

    @staticmethod
    def _dataset_table_identifier(dataset: FeatureDatasetDependency) -> str:
        """Return the Iceberg table identifier for one declared input dataset."""
        return IcebergTable.id_for(
            MaterializerConfig.iceberg_catalog(),
            dataset.dataset_type.iceberg_namespace,
            dataset.name,
            dataset.version,
        )

    @staticmethod
    def _planned_dataset(
        spark: SparkSession,
        output_partitions: list[MaterializerPartition],
        dataset: FeatureDatasetDependency,
    ) -> DataFrame:
        """Load exactly the snapshots and partitions pinned for one input dataset."""
        planned_partitions = {
            (
                dependency.partition_id,
                dependency.timestamp,
                dependency.snapshot_id,
            )
            for output_partition in output_partitions
            for dependency in output_partition.partition_dependencies
            if (dependency.dataset_name, dependency.dataset_version)
            == (dataset.name, dataset.version)
        }
        if not planned_partitions:
            raise DatasetPartitionPlanError(
                f"Materialization has no immutable partition plan for dataset {dataset.id}."
            )
        table_identifier = TransformMaterializer._dataset_table_identifier(
            dataset
        )
        partition_data_frames = [
            spark.read.option("versionAsOf", snapshot_id)
            .table(table_identifier)
            .where(
                spark_functions.to_date("timestamp")
                == spark_functions.to_date(
                    spark_functions.lit(partition_timestamp)
                )
            )
            for _, partition_timestamp, snapshot_id in sorted(
                planned_partitions,
                key=lambda partition: (partition[1], partition[0]),
            )
        ]
        data_frame = partition_data_frames[0]
        for partition_data_frame in partition_data_frames[1:]:
            data_frame = data_frame.unionByName(partition_data_frame)
        return data_frame.dropDuplicates()

    def _load_planned_datasets(self, spark: SparkSession) -> dict[str, Dataset]:
        """Load declared inputs under the transform argument names supplied by the API."""
        config = self.config
        injected_datasets: dict[str, Dataset] = {}
        for dataset in config.datasets:
            if dataset.argument_name in injected_datasets:
                raise DuplicateDatasetArgumentError(
                    "Declared datasets resolve to duplicate transform constructor argument "
                    f"{dataset.argument_name!r} for dataset {dataset.id}."
                )
            data_frame = self._planned_dataset(
                spark, config.partitions, dataset
            )
            injected_datasets[dataset.argument_name] = Dataset(
                metadata=DatasetReference(
                    dataset.name,
                    dataset.version,
                    dataset.num_lookback_partitions,
                ),
                frame=data_frame,
            )
        return injected_datasets

    @staticmethod
    def _is_transform_class[ConcreteTransform: DataTransform](
        candidate: object,
        transform_base: type[ConcreteTransform],
    ) -> TypeGuard[type[ConcreteTransform]]:
        """Return whether an imported object implements the requested transform contract."""
        return isinstance(candidate, type) and issubclass(
            candidate, transform_base
        )

    @staticmethod
    def _load_transform[ConcreteTransform: DataTransform](
        import_path: str,
        transform_base: type[ConcreteTransform],
    ) -> Callable[..., ConcreteTransform]:
        """Load a configured transform and preserve its verified concrete subtype."""
        module_name, class_name = import_path.rsplit(".", maxsplit=1)
        try:
            transform = import_module(module_name).__dict__[class_name]
        except (ImportError, KeyError) as error:
            raise TransformImportError(
                f"Cannot import configured transform {import_path!r}: {error}."
            ) from error
        if not TransformMaterializer._is_transform_class(
            transform, transform_base
        ):
            raise TransformTypeError(
                f"Configured transform {import_path!r} must inherit {transform_base.__name__}."
            )
        return transform

    def _inject_transform[ConcreteTransform: DataTransform](
        self,
        transform_class: Callable[..., ConcreteTransform],
        datasets: dict[str, Dataset],
    ) -> ConcreteTransform:
        """Construct a transform after verifying its API-declared dependencies and parameters."""
        config = self.config
        constructor_parameters = {
            name: parameter
            for name, parameter in signature(transform_class).parameters.items()
            if parameter.kind
            in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        }
        annotations = get_type_hints(transform_class.__init__)
        injected_parameters = {
            "keys": [FeatureKey(key) for key in config.feature_keys],
            "partition_type": config.partition_type,
            "data_type": DataType(config.dtype),
        }
        dataset_parameters = {
            name
            for name in constructor_parameters
            if annotations.get(name) is Dataset
        }
        if dataset_parameters != set(datasets):
            raise TransformDatasetInjectionError(
                "Transform dataset arguments differ from declared datasets "
                f"(constructor: {', '.join(sorted(dataset_parameters)) or 'none'}; "
                f"declared: {', '.join(sorted(datasets)) or 'none'})."
            )
        parameter_definitions = {
            name: parameter
            for name, parameter in constructor_parameters.items()
            if name not in dataset_parameters
            and name not in injected_parameters
        }
        required_parameters = {
            name
            for name, parameter in parameter_definitions.items()
            if parameter.default is Parameter.empty
        }
        missing_parameters = required_parameters.difference(config.parameters)
        unexpected_parameters = set(config.parameters).difference(
            parameter_definitions
        )
        if missing_parameters or unexpected_parameters:
            discrepancies = []
            if missing_parameters:
                discrepancies.append(
                    f"missing: {', '.join(sorted(missing_parameters))}"
                )
            if unexpected_parameters:
                discrepancies.append(
                    f"unexpected: {', '.join(sorted(unexpected_parameters))}"
                )
            raise TransformParameterError(
                f"Transform parameter names differ from the constructor ({'; '.join(discrepancies)})."
            )
        for name, value in config.parameters.items():
            expected_type = annotations.get(name)
            if (
                isinstance(expected_type, type)
                and type(value) is not expected_type
            ):
                raise TransformParameterError(
                    "Transform parameter type differs from the constructor contract "
                    f"(type for {name}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__})."
                )
        try:
            return transform_class(
                **datasets, **injected_parameters, **config.parameters
            )
        except TypeError as error:
            raise TransformParameterError(
                f"Cannot construct transform {transform_class.__name__}: {error}"
            ) from error

    def _transform(self, spark: SparkSession) -> Transform:
        """Load pinned datasets and instantiate this materializer's configured transform."""
        datasets = self._load_planned_datasets(spark)
        transform_class = self._load_transform(
            self.config.transform, self._transform_type
        )
        return self._inject_transform(transform_class, datasets)

    @property
    def _partition_type(self) -> PartitionType:
        """Return the configured output cadence for this transform resource."""
        return self.config.partition_type
