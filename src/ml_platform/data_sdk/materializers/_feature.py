"""Spark worker for one API-submitted feature materialization job."""
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
import json
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast, get_type_hints

from pydantic import Field, field_validator
from simple_classproperty import classproperty
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ml_platform.data_sdk.materializers._dataset import DatasetMaterializer
from ml_platform.data_sdk.materializers._base import Materializer, MaterializerConfig, MaterializerPartition, WriteMode
from ml_platform.data_sdk.materializers._materialization_validator import FeatureValidationInput, MaterializationValidator
from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._types import (
    DataType,
    Dataset,
    DatasetDependency,
    DatasetReference,
    FeatureDatasetDependency,
    FeatureKey,
    PartitionType,
)

_FEATURE_VALUE_COLUMN = "feature_value"
_DEPENDENCY_WRITE_LOCK = Lock()


class FeatureMaterializationConfig(MaterializerConfig):
    """Complete JSON contract consumed by a feature Spark materializer."""

    dtype: Literal["double", "float", "int", "bigint"] = Field(description="Spark SQL type used to persist feature values.")
    frequency: Literal["hourly", "daily"] = Field(description="Cadence of the feature's logical output partitions.")
    feature_keys: list[Literal["symbol"]] = Field(
        default_factory=lambda: ["symbol"], min_length=1,
        description="Ordered entity-key columns that identify each feature value.",
    )
    datasets: list[FeatureDatasetDependency] = Field(min_length=1, description="Versioned dataset dependencies mapped to transform arguments.")
    transform: str = Field(
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
        description="Absolute import path of the FeatureTransform implementation.",
    )
    parameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict, description="Typed keyword arguments passed to the feature transform constructor."
    )

    @field_validator("datasets")
    @classmethod
    def _unique_datasets(cls, datasets: list[FeatureDatasetDependency]) -> list[FeatureDatasetDependency]:
        if len({dataset.id for dataset in datasets}) != len(datasets):
            raise ValueError("Feature datasets must have unique name and version identities.")
        if len({dataset.argument_name for dataset in datasets}) != len(datasets):
            raise ValueError("Feature datasets must have unique transform argument names.")
        return datasets

    @field_validator("partitions")
    @classmethod
    def _require_partition_dependencies(cls, partitions: list[MaterializerPartition]) -> list[MaterializerPartition]:
        if any(not partition.partition_dependencies for partition in partitions):
            raise ValueError("Feature partitions must include pinned partition_dependencies.")
        return partitions

    @field_validator("feature_keys")
    @classmethod
    def _unique_feature_keys(cls, feature_keys: list[str]) -> list[str]:
        """Require an unambiguous feature row identity."""
        if len(set(feature_keys)) != len(feature_keys):
            raise ValueError("Feature keys must be unique.")
        return feature_keys




@dataclass(frozen=True)
class FeatureMaterializer(Materializer[FeatureMaterializationConfig]):
    """Materialize API-pinned feature partitions with their immutable inputs."""

    @classproperty
    def config_model(cls) -> type[FeatureMaterializationConfig]:
        """Return the strict runtime contract for feature jobs."""
        return FeatureMaterializationConfig

    @classproperty
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for feature jobs."""
        return "ml-platform-feature-materializer"

    @classproperty
    def iceberg_namespace(cls) -> str:
        """Store feature resources in their dedicated Iceberg namespace."""
        return "features"

    @classproperty
    def callback_resource(cls) -> str:
        """Report feature partition receipts as feature resources."""
        return "feature"

    @classproperty
    def write_mode(cls) -> WriteMode:
        """Replace only the feature partitions emitted by the transform."""
        return WriteMode.OVERWRITE

    def _partition_receipt_data(self, partition: MaterializerPartition) -> dict[str, object]:
        """Pin the exact dataset revisions consumed by this feature partition."""
        return {"dataset_partition_ids": [parent.partition_id for parent in partition.partition_dependencies]}

    @property
    def _feature_key_columns(self) -> tuple[str, ...]:
        """Return the immutable configured entity columns for one feature row."""
        return tuple(self.config.feature_keys)

    @classproperty
    def _dependency_manifest_table_identifier(cls) -> str:
        """Return the shared feature-to-dataset dependency manifest identifier."""
        return f"{cls.config_model.iceberg_catalog}.{cls.iceberg_namespace}.dependencies"

    @staticmethod
    def _planned_dataset(spark: Any, callbacks: list[MaterializerPartition], dataset: DatasetDependency) -> DataFrame:
        """Load only this dataset's pinned Iceberg revisions and day partitions."""
        planned_partitions = {
            (partition_dependency.partition_id, partition_dependency.timestamp, partition_dependency.snapshot_id)
            for callback in callbacks
            for partition_dependency in callback.partition_dependencies
            if (partition_dependency.dataset_name, partition_dependency.dataset_version) == (dataset.name, dataset.version)
        }
        if not planned_partitions:
            raise ValueError(f"Feature job has no immutable dataset partition plan for {dataset.id}.")
        table_identifier = FeatureMaterializer._dataset_table_identifier(dataset)
        partition_frames = [
            spark.read.option("versionAsOf", snapshot_id).table(table_identifier).where(
                F.to_date("timestamp") == F.to_date(F.lit(partition_timestamp))
            )
            for _, partition_timestamp, snapshot_id in sorted(
                planned_partitions, key=lambda partition: (partition[1], partition[0])
            )
        ]
        combined_partitions = partition_frames[0]
        for partition_frame in partition_frames[1:]:
            combined_partitions = combined_partitions.unionByName(partition_frame)
        return combined_partitions.dropDuplicates()

    @staticmethod
    def _materialize_definition(spark: SparkSession, config: FeatureMaterializationConfig) -> int:
        """Materialize one validated feature definition for local integration coverage."""
        for dataset in config.datasets:
            table = FeatureMaterializer._dataset_table_identifier(dataset)
            if not spark.catalog.tableExists(table):
                raise ValueError(f"Feature dataset table does not exist: {table}.")
        worker = FeatureMaterializer(config)
        layout_datasets = {
            dependency.argument_name: Dataset(
                metadata=DatasetReference(
                    dependency.name,
                    dependency.version,
                    dependency.num_lookback_partitions,
                ),
                frame=spark.table(FeatureMaterializer._dataset_table_identifier(dependency)),
            )
            for dependency in config.datasets
        }
        transform = worker._instantiate_feature_transform(layout_datasets)
        schema, partition_spec = worker._iceberg_layout(transform)
        worker._ensure_table(
            spark,
            schema=schema,
            partition_spec=partition_spec,
        )
        FeatureMaterializer._ensure_dependency_table(spark)
        _output_partitions = FeatureMaterializer._output_partitions(spark, config)
        dependencies = FeatureMaterializer._feature_dependencies(spark, config, _output_partitions)
        targets = FeatureMaterializer._target_output_partitions(spark, config, _output_partitions, dependencies).localCheckpoint(eager=True)
        if not targets.limit(1).count():
            return 0
        target_dependencies = dependencies.join(targets, "output_partition", "inner").localCheckpoint(eager=True)
        materialization_targets = target_dependencies.select("output_partition").distinct()
        if not materialization_targets.limit(1).count():
            FeatureMaterializer._delete_feature_partitions(spark, config, targets)
            FeatureMaterializer._replace_dependency_manifest(spark, config, target_dependencies, targets)
            return 0
        datasets = FeatureMaterializer._load_datasets(spark, config, target_dependencies)
        feature = worker._feature_transform(datasets)
        prepared_feature = worker._prepare_frame(feature)
        FeatureMaterializer._validate_feature_frame(
            prepared_feature, config, materialization_targets, datasets
        )
        row_count = prepared_feature.count()
        FeatureMaterializer._delete_feature_partitions(spark, config, targets)
        if row_count:
            (prepared_feature.withColumn("output_partition", FeatureMaterializer._partition_key(config))
                .repartition("output_partition")
                .sortWithinPartitions("output_partition", "timestamp", *config.feature_keys)
                .drop("output_partition")
                .writeTo(FeatureMaterializer._feature_table_identifier(config)).append())
        FeatureMaterializer._replace_dependency_manifest(spark, config, target_dependencies, targets)
        return row_count


    @staticmethod
    def _output_partitions(spark: SparkSession, config: FeatureMaterializationConfig) -> DataFrame:
        """Return all source and already-materialized output partition keys."""
        primary = FeatureMaterializer._primary_dataset(config)
        source = spark.table(FeatureMaterializer._dataset_table_identifier(primary)).select(
            FeatureMaterializer._partition_key(config).alias("output_partition")
        ).distinct()
        identifier = FeatureMaterializer._feature_table_identifier(config)
        if not spark.catalog.tableExists(identifier):
            return source
        completed = spark.table(identifier).select(FeatureMaterializer._partition_key(config).alias("output_partition")).distinct()
        return source.unionByName(completed).distinct()


    @staticmethod
    def _primary_dataset(config: FeatureMaterializationConfig) -> DatasetDependency:
        """Return the dataset that defines the feature's output partition domain."""
        return config.datasets[0]


    @staticmethod
    def _dataset_table_identifier(dataset: DatasetDependency) -> str:
        """Resolve a declared immutable dataset version to its physical table."""
        return DatasetMaterializer._iceberg_table_identifier(dataset.name, dataset.version)


    @staticmethod
    def _load_datasets(
        spark: SparkSession, config: FeatureMaterializationConfig, target_dependencies: DataFrame
    ) -> dict[str, Dataset]:
        """Load exactly the parent partitions declared for the target outputs."""
        datasets: dict[str, Dataset] = {}
        for dataset in config.datasets:
            if dataset.argument_name in datasets:
                raise ValueError(f"Feature datasets resolve to duplicate constructor argument: {dataset.argument_name}.")
            parent_partitions = target_dependencies.where(
                F.col("dataset_id") == dataset.id
            ).select("input_partition").distinct()
            datasets[dataset.argument_name] = Dataset(
                metadata=DatasetReference(dataset.name, dataset.version, dataset.num_lookback_partitions),
                frame=(
                    spark.table(FeatureMaterializer._dataset_table_identifier(dataset))
                    .withColumn("input_partition", F.date_trunc("day", F.col("timestamp")))
                    .join(parent_partitions, "input_partition", "inner")
                    .drop("input_partition")
                ),
            )
        return datasets


    @staticmethod
    def _feature_dependencies(
        spark: SparkSession, config: FeatureMaterializationConfig, _output_partitions: DataFrame
    ) -> DataFrame:
        """Hash every declared parent partition required by every output partition."""
        dependencies: DataFrame | None = None
        for dataset in config.datasets:
            parent_hashes = FeatureMaterializer._dataset_partition_hashes(spark, dataset)
            upper = F.date_trunc("day", F.col("output_partition"))
            lower = F.date_sub(upper, dataset.num_lookback_partitions)
            mapped = (
                _output_partitions.crossJoin(parent_hashes)
                .where((F.col("input_partition") >= lower) & (F.col("input_partition") <= upper))
                .withColumn("dataset_id", F.lit(dataset.id))
                .select("output_partition", "dataset_id", "input_partition", "content_hash")
            )
            dependencies = mapped if dependencies is None else dependencies.unionByName(mapped)
        assert dependencies is not None
        return dependencies


    @staticmethod
    def _dataset_partition_hashes(spark: SparkSession, dataset: DatasetDependency) -> DataFrame:
        """Compute deterministic content hashes for physical dataset partitions."""
        frame = spark.table(FeatureMaterializer._dataset_table_identifier(dataset))
        row = F.to_json(F.struct(*(F.col(column).alias(column) for column in sorted(frame.columns))))
        return (
            frame.withColumn("input_partition", F.date_trunc("day", F.col("timestamp")))
            .withColumn("row_hash", F.sha2(row, 256))
            .groupBy("input_partition")
            .agg(F.sha2(F.concat_ws("", F.sort_array(F.collect_list("row_hash"))), 256).alias("content_hash"))
        )


    @staticmethod
    def _target_output_partitions(
        spark: SparkSession,
        config: FeatureMaterializationConfig,
        _output_partitions: DataFrame,
        dependencies: DataFrame,
    ) -> DataFrame:
        """Return missing outputs plus existing outputs with changed dependency hashes."""
        identifier = FeatureMaterializer._feature_table_identifier(config)
        completed = spark.table(identifier).select(FeatureMaterializer._partition_key(config).alias("output_partition")).distinct()
        missing = _output_partitions.join(completed, "output_partition", "left_anti")
        stored = spark.table(FeatureMaterializer._dependency_manifest_table_identifier).where(
            (F.col("feature_name") == config.name) & (F.col("feature_version") == config.version)
        ).select("output_partition", "dataset_id", "input_partition", "content_hash")
        changed = dependencies.join(
            stored, ["output_partition", "dataset_id", "input_partition", "content_hash"], "left_anti"
        ).select("output_partition").unionByName(
            stored.join(
                dependencies, ["output_partition", "dataset_id", "input_partition", "content_hash"], "left_anti"
            ).select("output_partition")
        ).distinct()
        return missing.unionByName(changed).distinct()


    @staticmethod
    def _delete_feature_partitions(spark: SparkSession, config: FeatureMaterializationConfig, targets: DataFrame) -> None:
        """Delete stale output partitions before replacing them with recomputed rows."""
        identifier = FeatureMaterializer._feature_table_identifier(config)
        (
            targets.alias("source")
            .mergeInto(
                identifier,
                F.date_trunc(
                    FeatureMaterializer._spark_partition_unit(config), F.col(f"{identifier}.timestamp")
                ) == F.col("source.output_partition"),
            )
            .whenMatched()
            .delete()
            .merge()
        )


    @staticmethod
    def _ensure_dependency_table(spark: SparkSession) -> None:
        """Create the feature-output to dataset-partition dependency manifest."""
        identifier = FeatureMaterializer._dependency_manifest_table_identifier
        if spark.catalog.tableExists(identifier):
            return
        (
            spark.createDataFrame(
                [],
                "feature_name string, feature_version int, output_partition timestamp, "
                "dataset_id string, input_partition timestamp, content_hash string",
            )
            .writeTo(identifier)
            .using("iceberg")
            .tableProperty("format-version", "2")
            .tableProperty("write.format.default", "parquet")
            .partitionedBy("feature_name", "feature_version", "dataset_id")
            .create()
        )


    @staticmethod
    def _replace_dependency_manifest(
        spark: SparkSession, config: FeatureMaterializationConfig, dependencies: DataFrame, targets: DataFrame
    ) -> None:
        """Replace only the dependency records for output partitions just materialized."""
        with _DEPENDENCY_WRITE_LOCK:
            identifier = FeatureMaterializer._dependency_manifest_table_identifier
            replacement_targets = (
                targets.select("output_partition")
                .withColumn("feature_name", F.lit(config.name))
                .withColumn("feature_version", F.lit(config.version).cast("int"))
                .alias("source")
            )
            (
                replacement_targets.mergeInto(
                    identifier,
                    (F.col("source.feature_name") == F.col(f"{identifier}.feature_name"))
                    & (F.col("source.feature_version") == F.col(f"{identifier}.feature_version"))
                    & (F.col("source.output_partition") == F.col(f"{identifier}.output_partition")),
                )
                .whenMatched()
                .delete()
                .merge()
            )
            replacement = dependencies.withColumn("feature_name", F.lit(config.name)).withColumn(
                "feature_version", F.lit(config.version).cast("int")
            ).select(
                "feature_name", "feature_version", "output_partition", "dataset_id", "input_partition", "content_hash"
            )
            if replacement.limit(1).count():
                replacement.writeTo(FeatureMaterializer._dependency_manifest_table_identifier).append()

    @staticmethod
    def _partition_key(config: FeatureMaterializationConfig) -> F.Column:
        """Return the timestamp bucket used by the configured Iceberg partition transform."""
        partition_unit = {"hourly": "hour", "daily": "day"}[config.frequency]
        return F.date_trunc(partition_unit, F.col("timestamp"))


    @staticmethod
    def _load_transform(import_path: str) -> type[FeatureTransform]:
        """Load a FeatureTransform class referenced by its dotted import path."""
        module_name, class_name = import_path.rsplit(".", maxsplit=1)
        try:
            transform = getattr(import_module(module_name), class_name)
        except (AttributeError, ImportError) as error:
            raise ValueError(f"Cannot load feature transform: {import_path}") from error
        if not isinstance(transform, type) or not issubclass(transform, FeatureTransform):
            raise ValueError(f"Feature transform must inherit FeatureTransform: {import_path}")
        return cast(type[FeatureTransform], transform)


    def _instantiate_feature_transform(
        self,
        datasets: dict[str, Dataset],
    ) -> FeatureTransform:
        """Instantiate the configured transform with its declared datasets."""
        transform_class = self._load_transform(self.config.transform)
        parameters = self.config.parameters
        injected_parameters = {
            "keys": [FeatureKey(key) for key in self.config.feature_keys],
            "partition_type": PartitionType(self.config.frequency),
            "data_type": DataType(self.config.dtype),
        }
        constructor = signature(transform_class)
        constructor_parameters = {
            name: parameter
            for name, parameter in constructor.parameters.items()
            if parameter.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        }
        annotations = get_type_hints(transform_class.__init__)
        dataset_parameters = {
            name for name in constructor_parameters if annotations.get(name) is Dataset
        }
        if dataset_parameters != set(datasets):
            raise ValueError(
                "Feature transform dataset discrepancy "
                f"(constructor: {', '.join(sorted(dataset_parameters)) or 'none'}; "
                f"declared: {', '.join(sorted(datasets)) or 'none'})."
            )
        parameter_definitions = {
            name: parameter
            for name, parameter in constructor_parameters.items()
            if name not in dataset_parameters and name not in injected_parameters
        }
        required = {
            name for name, parameter in constructor_parameters.items()
            if name not in dataset_parameters and name not in injected_parameters and parameter.default is Parameter.empty
        }
        missing = required.difference(parameters)
        unexpected = set(parameters).difference(parameter_definitions)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if unexpected:
                details.append(f"unexpected: {', '.join(sorted(unexpected))}")
            raise ValueError(f"Feature transform parameter discrepancy ({'; '.join(details)}).")
        for name, value in parameters.items():
            expected_type = annotations.get(name)
            if isinstance(expected_type, type) and type(value) is not expected_type:
                raise ValueError(
                    f"Feature transform parameter discrepancy (type for {name}: "
                    f"expected {expected_type.__name__}, got {type(value).__name__})."
                )
        try:
            transform = transform_class(**datasets, **injected_parameters, **parameters)
        except TypeError as error:
            raise ValueError(
                f"Feature transform parameter discrepancy for {transform_class.__name__}: {error}"
            ) from error
        return transform

    def _feature_transform(self, datasets: dict[str, Dataset]) -> DataFrame:
        """Run the configured transform for local feature integration callers."""
        return self._instantiate_feature_transform(datasets)()


    @staticmethod
    def _validate_feature_frame(
        frame: DataFrame,
        config: FeatureMaterializationConfig,
        requested_partitions: DataFrame,
        datasets: dict[str, Dataset],
    ) -> None:
        """Validate one prepared feature DataFrame before replacing its partitions."""
        feature_keys = tuple(config.feature_keys)
        expected_schema = (
            "timestamp TIMESTAMP, "
            f"{', '.join(f'{key} STRING' for key in feature_keys)}, "
            f"{_FEATURE_VALUE_COLUMN} {config.dtype.upper()}"
        )
        expected_partition_spec = {"hourly": "hours(timestamp)", "daily": "days(timestamp)"}[config.frequency]
        partitioned = frame.withColumn("output_partition", FeatureMaterializer._partition_key(config))
        partition_interval = FeatureMaterializer(config)._partition_interval(expected_partition_spec)
        MaterializationValidator(config.validations).validate(
            FeatureValidationInput(
                frame=frame,
                partitioned_frame=partitioned,
                requested_partitions=requested_partitions,
                partition_column="output_partition",
                partition_interval=partition_interval,
                exact_columns=frozenset({"timestamp", *feature_keys, _FEATURE_VALUE_COLUMN}),
                non_null_columns=("timestamp", *feature_keys, _FEATURE_VALUE_COLUMN),
                unique_key_columns=("timestamp", *feature_keys),
                feature_value_column=_FEATURE_VALUE_COLUMN,
                feature_keys=feature_keys,
                datasets=datasets,
                actual_schema=expected_schema,
                expected_schema=expected_schema,
                actual_partition_spec=expected_partition_spec,
                expected_partition_spec=expected_partition_spec,
            )
        )


    @staticmethod
    def _feature_table_identifier(config: FeatureMaterializationConfig) -> str:
        """Return an Iceberg identifier unique to the feature name and version."""
        return FeatureMaterializer._iceberg_table_identifier(config.name, config.version)


    @staticmethod
    def _feature_storage_path(warehouse_dir: str | Path, config: FeatureMaterializationConfig) -> Path:
        """Return the versioned feature directory under the configured data root."""
        table_name = FeatureMaterializer.config_model._iceberg_table_name_for(config.name, config.version)
        return Path(warehouse_dir).resolve() / FeatureMaterializer.iceberg_namespace / table_name


    def _additional_table_properties(self) -> dict[str, str]:
        """Add feature-specific fields to the shared resource table metadata."""
        return {
            "feature.dtype": self.config.dtype,
            "feature.feature_keys": json.dumps(self.config.feature_keys),
            "feature.transform": self.config.transform,
        }

    def _requested_partitions(self, spark: SparkSession, partition_spec: str) -> DataFrame:
        """Represent the requested output partitions for Spark validation and filtering."""
        rows = [
            (self._partition_timestamp(partition.timestamp, partition_spec),)
            for partition in self.config.partitions
        ]
        return spark.createDataFrame(rows, "output_partition timestamp").distinct()

    def _load_planned_datasets(self, spark: SparkSession) -> dict[str, Dataset]:
        """Inject every declared dataset under its explicit constructor argument name."""
        declared_ids = {(dataset.name, dataset.version) for dataset in self.config.datasets}
        planned_ids = {
            (partition_dependency.dataset_name, partition_dependency.dataset_version)
            for callback in self.config.partitions
            for partition_dependency in callback.partition_dependencies
        }
        if planned_ids != declared_ids:
            raise ValueError("Feature dataset partition plan does not match declared datasets.")

        datasets: dict[str, Dataset] = {}
        for dependency in self.config.datasets:
            parent_frame = self._planned_dataset(spark, self.config.partitions, dependency)
            reference = DatasetReference(
                dependency.name, dependency.version, dependency.num_lookback_partitions
            )
            datasets[dependency.argument_name] = Dataset(metadata=reference, frame=parent_frame)
        return datasets

    def _filter_partitions(self, frame: DataFrame, requested: DataFrame, partition_spec: str) -> DataFrame:
        """Retain transform output belonging to the requested logical partitions."""
        return (
            frame.withColumn(
                "output_partition",
                F.date_trunc(self._partition_unit(partition_spec), F.col("timestamp")),
            )
            .join(requested, "output_partition", "left_semi")
            .drop("output_partition")
        )

    def _row_digest(self) -> F.Column:
        """Preserve the feature content hash format used by existing revisions."""
        value_columns = ("timestamp", *self._feature_key_columns, _FEATURE_VALUE_COLUMN)
        serialized_values = [F.coalesce(F.col(column).cast("string"), F.lit("")) for column in value_columns]
        return F.sha2(F.concat_ws("\u0001", *serialized_values), 256)

    def _prepare_frame(self, frame: DataFrame) -> DataFrame:
        """Cast transform values to the feature's persisted schema."""
        return frame.select(
            F.col("timestamp").try_cast("timestamp").alias("timestamp"),
            *(F.col(key).cast("string").alias(key) for key in self._feature_key_columns),
            F.col(_FEATURE_VALUE_COLUMN).try_cast(self.config.dtype).alias(_FEATURE_VALUE_COLUMN),
        )

    def _create_transform(self, spark: SparkSession) -> FeatureTransform:
        """Build the feature transform from the pinned dataset revisions."""
        datasets = self._load_planned_datasets(spark)
        transform = self._instantiate_feature_transform(datasets)
        return transform

    def validate(self, frame: DataFrame, spark: SparkSession, transform: FeatureTransform) -> None:
        """Validate typed feature values, partitions, input keys, and transform layout."""
        partition_spec = transform.iceberg_partition_spec
        requested_partitions = self._requested_partitions(spark, partition_spec)
        input_datasets = self._load_planned_datasets(spark)
        expected_schema = (
            "timestamp TIMESTAMP, "
            f"{', '.join(f'{key} STRING' for key in self._feature_key_columns)}, "
            f"{_FEATURE_VALUE_COLUMN} {self.config.dtype.upper()}"
        )
        expected_partition_spec = {"hourly": "hours(timestamp)", "daily": "days(timestamp)"}[self.config.frequency]
        partitioned = frame.withColumn("output_partition", F.date_trunc(self._partition_unit(partition_spec), F.col("timestamp")))
        self._validator.validate(
            FeatureValidationInput(
                frame=frame,
                partitioned_frame=partitioned,
                requested_partitions=requested_partitions,
                partition_column="output_partition",
                partition_interval=self._partition_interval(partition_spec),
                exact_columns=frozenset({"timestamp", *self._feature_key_columns, _FEATURE_VALUE_COLUMN}),
                non_null_columns=("timestamp", *self._feature_key_columns, _FEATURE_VALUE_COLUMN),
                unique_key_columns=("timestamp", *self._feature_key_columns),
                feature_value_column=_FEATURE_VALUE_COLUMN,
                feature_keys=self._feature_key_columns,
                datasets=input_datasets,
                actual_schema=transform.iceberg_schema,
                expected_schema=expected_schema,
                actual_partition_spec=transform.iceberg_partition_spec,
                expected_partition_spec=expected_partition_spec,
            )
        )



    def materialize(self, spark: SparkSession, transform: FeatureTransform) -> DataFrame:
        """Produce only the requested feature partitions."""
        partition_spec = transform.iceberg_partition_spec
        requested_partitions = self._requested_partitions(spark, partition_spec)
        feature = transform()
        filtered_feature = self._filter_partitions(feature, requested_partitions, partition_spec)
        return filtered_feature


if __name__ == "__main__":
    FeatureMaterializer.main()
